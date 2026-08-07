"""The single JüNA agent for neutron science and repository research.

Repository investigation uses embedding search over local code and
documentation when a healthy index is available. Registration happens at
import time via :func:`register_agent_factory`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langgraph.graph.state import CompiledStateGraph

from juena.core.llms_providers import (
    build_chat_model,
    get_available_providers,
    get_default_model,
)
from juena.core.config import global_config
from juena.core.log import get_logger
from juena.agents.skills_backend import (
    JUENA_MEMORY_SYSTEM_PROMPT,
    MEMORY_SOURCES,
    ReadOnlyFilesystemBackend,
    ReadOnlyInputsStateBackend,
    build_user_context_backend,
)
from juena.indexing.bootstrap import describe_bootstrap_problems
from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.schema.llm_models import BlabladorModelName, Provider
from juena.server.agent_registry import register_agent_alias, register_agent_factory
from juena.server.checkpointer import get_checkpointer
from juena.server.store import get_store
from juena.server.runtime_model_middleware import (
    RuntimeModelContext,
    RuntimeModelMiddleware,
)
from juena.tools.context7 import Context7Runtime, load_optional_context7_tools
from juena.tools.tavily import load_optional_tavily_tool
from juena.tools.repo_search import (
    build_code_chat_tools,
)

logger = get_logger(__name__)

# The single JüNA agent. Older ids stay registered as aliases in the registry.
JUENA_AGENT_ID = "juena"

# Loop bounds. The graph also carries recursion_limit=1000, but that is a
# blunt backstop; these fail in a way the user can read.
COORDINATOR_MODEL_CALL_LIMIT = 50
COORDINATOR_TOOL_CALL_LIMIT = 100
SUBAGENT_MODEL_CALL_LIMIT = 60
SUBAGENT_TOOL_CALL_LIMIT = 150

# Transient-failure handling for model and tool calls.
MODEL_MAX_RETRIES = 3
TOOL_MAX_RETRIES = 2

CODE_CHAT_STATIC_PROVIDER = Provider.BLABLADOR.value
CODE_CHAT_SUBAGENT_MODEL = BlabladorModelName.MINIMAX_M27.value
CODE_CHAT_SUMMARIZER_MODEL = BlabladorModelName.GPT_OSS.value
CODE_CHAT_RESEARCH_SUBAGENT_NAME = "code-chat-expert"
CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List an absolute directory path.\n\n"
        "Code-chat policy: indexed search is the primary repository discovery "
        "layer. Use ls('/repos') only when the user asks which repositories "
        "exist or indexed search still leaves the repo id unclear. Use ls on a "
        "narrow directory only after indexed search or read_file identifies "
        "that directory as relevant."
    ),
    "read_file": (
        "Read an absolute file path with optional offset and limit parameters. "
        "Results include line numbers.\n\n"
        "Code-chat policy: use read_file for exact evidence from the few best "
        "indexed search hits, or for staged /inputs files. Do not use repeated "
        "read_file calls as broad codebase search; refine indexed search first."
    ),
    "grep": (
        "Search for literal text across files.\n\n"
        "Code-chat policy: indexed search first. Grep is exact verification, "
        "not semantic discovery. Use grep only for exact tokens, symbols, "
        "config keys, environment variables, imports, or call sites after "
        "indexed search identifies a likely repo, file, or area, or when the "
        "user query is itself an exact token. Do not use grep for broad "
        "repository crawling."
    ),
    "glob": (
        "Find files matching a glob pattern.\n\n"
        "Code-chat policy: use glob only for precise filename, extension, test, "
        "docs, or generated-config patterns after indexed search suggests the "
        "relevant area, or when the user asks for matching file names. Do not "
        "use glob as first-pass repository exploration."
    ),
}

# The coordinator's filesystem tools exist to serve skills and memory, not
# repository research. Its backend does not route /repos at all, and staged
# /inputs stay read-only, but say so in the tool surface too so the model does
# not try to investigate the codebase itself instead of delegating.
_COORDINATOR_DELEGATION_NOTE = (
    "Coordinator policy: this tool reaches only the global skills library "
    "(`/skills/<name>/SKILL.md`) and your per-user memory "
    "(`/memories/AGENTS.md`). Repository files, staged `/inputs`, and all code "
    "reading belong to the `code-chat-expert` subagent via the `task` tool - "
    "delegate those instead of attempting them here."
)
CODE_CHAT_COORDINATOR_FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": f"List an absolute directory path.\n\n{_COORDINATOR_DELEGATION_NOTE}",
    "read_file": (
        "Read an absolute file path with optional offset and limit "
        f"parameters.\n\n{_COORDINATOR_DELEGATION_NOTE} Use it to open a skill "
        "whose description matches the task."
    ),
    "grep": f"Search for literal text across files.\n\n{_COORDINATOR_DELEGATION_NOTE}",
    "glob": f"Find files matching a glob pattern.\n\n{_COORDINATOR_DELEGATION_NOTE}",
    "write_file": (
        "Create or overwrite a file.\n\nCoordinator policy: only "
        "`/memories/AGENTS.md`, and only when the user explicitly asks you to "
        "remember something. Skills are read-only."
    ),
    "edit_file": (
        "Perform an exact string replacement in a file.\n\nCoordinator policy: "
        "only `/memories/AGENTS.md`, and only when the user explicitly asks "
        "you to remember or forget something. Skills are read-only."
    ),
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUENA_IDENTITY = """\
You are JüNA, an assistant for neutron science and research software.

You answer scientific questions and help with software engineering.\
"""

# Appended to the identity only when a repository index is attached.
_IDENTITY_WITH_REPOS = (
    " You also answer questions about the software repositories indexed into"
    " this deployment."
)

# Rules that hold no matter which tools loaded.
_RULES_CORE: tuple[str, ...] = (
    "Keep the thread focused on the user's request: create a todo list for "
    "multi-step work, do the work, then give one synthesized answer.",
    "If the user's message is just a greeting or small talk, respond directly "
    "without delegating or calling tools.",
    "Do not expose internal scratch notes, todos, or subagent mechanics unless "
    "the user explicitly asks for them.",
)

# Only meaningful when the repo-research subagent is attached.
_RULES_REPO: tuple[str, ...] = (
    "You are the coordinator, not the repository-file expert. You know indexed "
    "local repositories exist, but you do not know their contents in advance.",
    "For every repository question, code-reading task, staged-input inspection, "
    "or request about how the local neutron-science or experiment software "
    f"works, delegate first to the `{CODE_CHAT_RESEARCH_SUBAGENT_NAME}` "
    "subagent via the `task` tool.",
    "When delegating repository investigation, explicitly ask the subagent to "
    'use indexed search first: `search_code_hybrid` with `repo_id="all"` when '
    "the relevant repository is unknown, `search_code_semantic` or "
    "`search_docs_local` when they better match the task, and filesystem tools "
    "only to verify the few best indexed hits.",
    "Treat subagent returns as structured research packets. Compare packets, "
    "reconcile conflicts, remove duplication, and write the user-facing "
    "synthesis yourself.",
    "Do not answer repository or codebase questions from model memory. Wait for "
    "cited evidence before answering.",
    "Indexed search previews are leads, not evidence. If a subagent packet only "
    "cites search previews or lacks `read_file` line-number evidence for an "
    "important local claim, delegate a narrower follow-up or clearly mark the "
    "gap.",
    "Cite file paths with line numbers whenever the subagent provides enough "
    "local evidence.",
)

# Used instead of _RULES_REPO when no repository index is available, so the
# model does not offer to search repositories it cannot reach.
_RULES_NO_REPO: tuple[str, ...] = (
    "No local repository index is available in this deployment. If the user "
    "asks about local repository contents, say the repository index is "
    "unavailable rather than guessing at file paths or code.",
)

_RULES_CONTEXT7: tuple[str, ...] = (
    "Use Context7 for upstream library, framework, and dependency "
    "documentation. Resolve the library first, then fetch the relevant docs.",
    "If the user explicitly says `use context7`, you must call an available "
    "Context7 tool before answering.",
    "Label Context7 claims as external documentation and name the library.",
)

_RULES_TAVILY: tuple[str, ...] = (
    "Use Tavily web search for current events, live facts, and external "
    "questions that local evidence and documentation do not cover. Prefer "
    "Tavily-grounded answers over unverified model memory for factual claims "
    "about the outside world.",
    "Start with one precise search query. Only search again if the first "
    "search is insufficient. Do not run broad or repetitive search loops.",
    "Label Tavily claims as web search and include source links.",
)

_RULE_CITE_EVIDENCE = (
    "Every substantive claim must be grounded in cited evidence -- {sources}. "
    "Never present unsupported model memory as fact."
)

_RULE_STATE_UNCERTAINTY = (
    "You have no retrieval tools in this deployment, so answer from your own "
    "knowledge and say plainly when you are uncertain or when a claim would "
    "need checking against a source."
)

_RULES_OUTPUT: tuple[str, ...] = (
    "Your final response must be concise and directly answer the question.",
)


def build_system_prompt(
    *,
    has_repo_research: bool,
    has_context7: bool,
    has_tavily: bool,
) -> str:
    """Compose the system prompt from the capabilities actually available.

    Sections for absent tools are omitted entirely: instructing the model to use
    a tool it has not been given produces refusals and hallucinated tool names.
    """
    rules: list[str] = []
    rules.extend(_RULES_CORE)
    rules.extend(_RULES_REPO if has_repo_research else _RULES_NO_REPO)
    if has_context7:
        rules.extend(_RULES_CONTEXT7)
    if has_tavily:
        rules.extend(_RULES_TAVILY)

    evidence_sources = [
        source
        for source, available in (
            ("indexed local repositories", has_repo_research),
            ("Context7 documentation", has_context7),
            ("Tavily web search", has_tavily),
        )
        if available
    ]
    if evidence_sources:
        if len(evidence_sources) > 1:
            rules.append(
                "Prefer evidence in this order: " + ", then ".join(evidence_sources) + "."
            )
        rules.append(_RULE_CITE_EVIDENCE.format(sources=" or ".join(evidence_sources)))
    else:
        # With no retrieval at all, demanding citations just invites invented ones.
        rules.append(_RULE_STATE_UNCERTAINTY)

    rules.extend(_RULES_OUTPUT)

    identity = JUENA_IDENTITY + (_IDENTITY_WITH_REPOS if has_repo_research else "")
    numbered = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1))
    return f"{identity}\n\n## Rules\n\n{numbered}\n"

CODE_CHAT_RESEARCH_SUBAGENT_PROMPT = """\
You are the repository expert for indexed neutron-science and experiment
software repositories.

## Required workflow

1. Always search first. Do not answer from memory.
2. Use indexed search as the primary repository navigation layer:
   - Use `search_code_hybrid` by default for repository questions, semantic
     concepts, symbol names, filenames, and config keys.
   - Use `repo_id="all"` when the relevant repository is unknown, when several
     repositories may matter, or when you would otherwise need `ls("/repos")`
     only to choose a repo.
   - Use a specific repo id when the user names one or indexed results already
     identify the relevant repository.
   - Use `search_code_semantic` for conceptual follow-ups when hybrid results
     are too broad or miss the user's intent.
   - Use `search_docs_local` for README, docs, setup, usage, or guide-focused
     questions.
3. Treat indexed search results as candidate paths and compact previews, not
   final evidence.
4. Use filesystem tools sparingly and only after indexed search identifies
   likely files or symbols:
   - Use `read_file` on the few best `/repos/...` paths needed for exact
     line-numbered evidence.
   - Use `grep` only to follow exact symbols, class/function names, config
     keys, imports, or call sites discovered from the user request or indexed
     hits.
   - Use `glob` only for precise filename patterns, tests, examples, docs, or
     generated config locations suggested by indexed results.
   - Use `ls("/repos")` only when the user asks what repositories are
     available or when indexed search still leaves the repo id unclear.
5. Do not use filesystem tools as broad search or repository crawling. Prefer
   more indexed search refinement over broad `ls`, `grep`, `glob`, or repeated
   `read_file` calls.
6. If `/inputs/` exists, inspect it before or alongside repository search when
   the user's question depends on staged files. Treat
   `/inputs/uploads/...` as persistent chat uploads and `/inputs/current_*` as
   turn-scoped helper files for the latest message.
7. If present, inspect `/inputs/uploads_manifest.md` to discover the full set
   of uploads available in this chat.
8. For large staged user files, page through `/inputs/...` with `read_file`
   offsets and use `grep` before reading more.
9. Exhaust local repository evidence before consulting external sources.
10. Use Context7 tools next for upstream libraries, frameworks, and dependency
   docs/examples. Resolve the library first, then fetch the relevant docs.
11. Tavily web search is not available in this subagent. Stay with local
   repository tools and Context7.
12. If the user explicitly says `use context7`, you must call an available
    Context7 tool before answering. If no Context7 tool is available, say that
    clearly.
13. Use Context7 to enrich local repository findings when the answer depends on
    upstream framework behavior, external APIs, or dependency docs that are not
    fully explained in the configured repository itself.
14. Search results from local repo tools include a canonical `/repos/...` path.
    When a result looks relevant, read that path with `read_file` for exact
    evidence.
15. Prefer local repository evidence over Context7 whenever the answer depends
    on configured repository behavior or implementation details.
16. Never invent local file paths or line numbers for Context7 results. Cite
    Context7 findings as external documentation and name the library when you
    use them.
17. Do not broaden the investigation with repetitive searches. Use one focused
    repo or Context7 lookup at a time and only refine when needed.
18. Never write to or edit files under `/repos`. Those paths are read-only.
19. Ask a concise follow-up only when the staged user inputs and repository
    evidence are still insufficient for a useful answer.
20. Keep any scratch files short and disposable. Use them only to compress your
    own working notes, not as a substitute for evidence.
21. Never write to or edit files under `/inputs`. Those paths are read-only
    staged user evidence. When suggesting a fix, return the changed code or
    patch in the response instead of editing staged files.
22. When the answer depends on external library behavior and the relevant files
    are read-only, prefer using available Context7 tools to ground the
    recommendation before you answer.
23. If the remaining gap is outside local repositories and outside Context7's
    scope, say so clearly so the coordinator can use Tavily web search as a
    last resort.

## Output requirements

- Return a structured research packet to the coordinator, not a final
  user-facing answer or narrative essay.
- Preserve evidence rather than over-compressing it. The coordinator will do
  the final synthesis across one or more packets.
- Use these sections:
  1. Direct finding: the shortest accurate answer to the delegated task.
  2. Evidence ledger: every important local claim with supporting `/repos/...`
     file paths, line numbers, and a brief explanation. Use line-numbered
     `read_file` evidence here, not indexed-search previews.
  3. Relevant code flow: the function, class, module, or configuration
     relationships needed to understand the answer.
  4. Ambiguities or gaps: what could not be verified, conflicting evidence, or
     missing context.
  5. Recommended final answer points: concise points the coordinator should
     include when answering the user.
- For Context7-based claims, say they come from external documentation and name
  the library or framework you used. Do not invent local line numbers for them.
- If the evidence is ambiguous or incomplete, say so clearly.
"""


@dataclass
class CodeChatAgentResources:
    """Keep long-lived resources alive for the cached agent instance."""

    app: CompiledStateGraph
    context7_runtime: Context7Runtime | None = None


def _build_code_chat_backend(
    repo_cache_root: Path,
) -> CompositeBackend:
    """Create a Deep Agents backend with scratch state + read-only repos."""

    repos_backend = ReadOnlyFilesystemBackend(
        root_dir=repo_cache_root,
        virtual_mode=True,
        label="read-only repository",
    )
    return CompositeBackend(
        default=ReadOnlyInputsStateBackend(),
        routes={
            "/repos/": repos_backend,
        },
    )


def build_fallback_models() -> list[Any]:
    """Chat models to try when the primary provider fails at call time.

    The coordinator's primary model is selected per request, while its graph is
    cached process-wide.  Therefore fallback construction must depend only on
    deployment configuration, not on whichever provider happened to create the
    cached graph first.
    """
    fallback_provider = (global_config.FALLBACK_PROVIDER or "").strip().lower()
    if not fallback_provider:
        return []

    if not get_available_providers().get(fallback_provider, False):
        logger.warning(
            "FALLBACK_PROVIDER '%s' is not configured; running without a model fallback",
            fallback_provider,
        )
        return []

    fallback_model = get_default_model(fallback_provider)
    if not fallback_model:
        return []

    logger.info("Model fallback enabled: %s/%s", fallback_provider, fallback_model)
    return [build_chat_model(provider=fallback_provider, model=fallback_model)]


def _resilience_middleware(
    fallback_models: list[Any],
    *,
    model_call_limit: int,
    tool_call_limit: int,
) -> list[Any]:
    """Retry/fallback/limit middleware, in nesting order.

    Ordering matters. ``wrap_*`` hooks nest with the first entry outermost, and
    callers place these *after* ``RuntimeModelMiddleware`` so the per-request
    model override happens outside the fallback. If the fallback were outer, the
    runtime override would reinstate the failed model on every retry and loop.

    Fallback sits outside retry so a model is retried for transient errors
    first, and only swapped once those retries are exhausted.
    """
    middleware: list[Any] = []
    if fallback_models:
        middleware.append(ModelFallbackMiddleware(*fallback_models))
    middleware.extend([
        ModelRetryMiddleware(
            max_retries=MODEL_MAX_RETRIES,
            # The fallback middleware can only try its alternative model when
            # retry exhaustion remains an exception. LangChain's default
            # "continue" mode converts it into a successful ModelResponse.
            on_failure="error",
        ),
        ToolRetryMiddleware(max_retries=TOOL_MAX_RETRIES),
        ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="continue"),
    ])
    return middleware


def _build_repo_research_subagent(
    llm_subagent: Any,
    llm_summarizer: Any,
    tools: list[Any],
    backend: Any,
    fallback_models: list[Any] | None = None,
) -> CompiledSubAgent:
    """Create the repo-analysis subagent with bound retrieval tools."""

    subagent = create_agent(
        model=llm_subagent,
        tools=tools,
        system_prompt=CODE_CHAT_RESEARCH_SUBAGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS,
            ),
            create_summarization_middleware(llm_summarizer, backend),
            PatchToolCallsMiddleware(),
            *_resilience_middleware(
                fallback_models or [],
                model_call_limit=SUBAGENT_MODEL_CALL_LIMIT,
                tool_call_limit=SUBAGENT_TOOL_CALL_LIMIT,
            ),
        ],
        context_schema=RuntimeModelContext,
        name="code_chat_repo_research_subagent",
    )
    return {
        "name": CODE_CHAT_RESEARCH_SUBAGENT_NAME,
        "description": (
            "Use this agent for repository analysis, staged input inspection, "
            "code search, file reading, and multi-step questions about indexed "
            "repositories. Use indexed search first: `search_code_hybrid` with "
            '`repo_id="all"` when the repo is unknown, `search_code_semantic` '
            "for conceptual follow-ups, and `search_docs_local` for docs. Use "
            "filesystem tools sparingly for line-numbered verification: "
            "`read_file` for the few best hits, `grep`/`glob` only for exact "
            "symbols or file patterns, and `ls('/repos')` only when needed. "
            "Use Context7 only for external dependencies or upstream "
            "frameworks. This subagent does not use Tavily web search. Return "
            "a structured research packet with evidence, not a final "
            "user-facing answer."
        ),
        "runnable": subagent,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_juena_agent(
    provider: str,
    model: str,
) -> tuple[CodeChatAgentResources, CompiledStateGraph]:
    """Factory called by the agent registry on first use."""

    repo_manager = RepoManager()
    vector_index = RepoVectorIndex(repo_manager)
    code_chat_backend = _build_code_chat_backend(repo_manager.cache_dir)
    store = get_store()
    coordinator_backend = build_user_context_backend(global_config.SKILLS_DIR, store)

    # Bootstrap is expected to happen before the server starts, but a missing
    # index must not take down the whole agent: JüNA also answers science
    # questions that never touch a repository. Degrade to no repo research.
    bootstrap_problems = describe_bootstrap_problems(repo_manager, vector_index)
    has_repo_research = not bootstrap_problems
    if bootstrap_problems:
        logger.warning(
            "Repository research disabled: %s. Run `python -m juena.indexing.bootstrap` "
            "to enable it.",
            "; ".join(bootstrap_problems),
        )

    llm = build_chat_model(provider=provider, model=model)
    llm_summarizer = build_chat_model(
        provider=CODE_CHAT_STATIC_PROVIDER,
        model=CODE_CHAT_SUMMARIZER_MODEL,
        temperature=0.0,
    )
    context7_runtime = await load_optional_context7_tools()
    context7_tools = list(context7_runtime.tools) if context7_runtime is not None else []
    tavily_tool = load_optional_tavily_tool()
    coordinator_tools = list(context7_tools)
    if tavily_tool is not None:
        coordinator_tools.append(tavily_tool)

    fallback_models = build_fallback_models()

    subagents = []
    if has_repo_research:
        llm_subagent = build_chat_model(
            provider=CODE_CHAT_STATIC_PROVIDER,
            model=CODE_CHAT_SUBAGENT_MODEL,
            temperature=0.0,
        )
        repo_subagent_tools = list(build_code_chat_tools(repo_manager, vector_index))
        repo_subagent_tools.extend(context7_tools)
        subagents.append(
            _build_repo_research_subagent(
                llm_subagent,
                llm_summarizer,
                repo_subagent_tools,
                code_chat_backend,
                fallback_models=build_fallback_models(),
            )
        )

    middleware: list[Any] = [
        TodoListMiddleware(),
        SkillsMiddleware(
            backend=coordinator_backend,
            sources=[("/skills/", "Global")],
        ),
        FilesystemMiddleware(
            backend=coordinator_backend,
            custom_tool_descriptions=CODE_CHAT_COORDINATOR_FILESYSTEM_TOOL_DESCRIPTIONS,
        ),
        MemoryMiddleware(
            backend=coordinator_backend,
            sources=MEMORY_SOURCES,
            system_prompt=JUENA_MEMORY_SYSTEM_PROMPT,
        ),
    ]
    if subagents:
        middleware.append(
            SubAgentMiddleware(backend=code_chat_backend, subagents=subagents)
        )
    middleware.extend([
        create_summarization_middleware(llm_summarizer, code_chat_backend),
        PatchToolCallsMiddleware(),
        # RuntimeModelMiddleware must precede the resilience middleware: wrap
        # hooks nest first-outermost, so the per-request model override has to
        # happen outside the fallback. Reversed, the override would reinstate
        # the failed model on every fallback attempt.
        RuntimeModelMiddleware(),
        *_resilience_middleware(
            fallback_models,
            model_call_limit=COORDINATOR_MODEL_CALL_LIMIT,
            tool_call_limit=COORDINATOR_TOOL_CALL_LIMIT,
        ),
    ])

    agent = create_agent(
        model=llm,
        tools=coordinator_tools,
        system_prompt=build_system_prompt(
            has_repo_research=has_repo_research,
            has_context7=bool(context7_tools),
            has_tavily=tavily_tool is not None,
        ),
        middleware=middleware,
        # No explicit state_schema: FilesystemMiddleware already declares
        # FilesystemState and create_agent merges middleware state schemas.
        context_schema=RuntimeModelContext,
        checkpointer=get_checkpointer(),
        store=store,
        name=JUENA_AGENT_ID,
    ).with_config({"recursion_limit": 1000})

    resources = CodeChatAgentResources(
        app=agent,
        context7_runtime=context7_runtime,
    )
    return (resources, agent)


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_agent_factory(JUENA_AGENT_ID, create_juena_agent, set_as_default=True)

# Legacy ids kept resolvable for one release so in-flight browser state and
# bookmarked links do not 404 on deploy.
register_agent_alias("code_chat_agent", JUENA_AGENT_ID)
register_agent_alias("react_agent", JUENA_AGENT_ID)
