"""
Code-Chat Agent – answers questions about software repositories using
hybrid (keyword + embedding) search over local code and documentation.

Registration happens at import time via ``register_agent_factory``.
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import ToolRuntime
from langgraph.graph.state import CompiledStateGraph

from juena.core.llms_providers import LLMFactory, create_llm_with_fallback
from juena.indexing.bootstrap import validate_bootstrap_ready
from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.schema.llm_models import BlabladorModelName, Provider
from juena.server.agent_registry import register_agent_factory
from juena.server.checkpointer import get_checkpointer
from juena.server.runtime_model_middleware import (
    RuntimeModelContext,
    RuntimeModelMiddleware,
)
from juena.tools.context7 import Context7Runtime, load_optional_context7_tools
from juena.tools.tavily import load_optional_tavily_tool
from juena.tools.repo_search import (
    build_code_chat_tools,
)

CODE_CHAT_STATIC_PROVIDER = Provider.BLABLADOR.value
CODE_CHAT_SUBAGENT_MODEL = "alias-code"
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

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CODE_CHAT_SYSTEM_PROMPT = """\
You are the code-chat coordinator for indexed neutron-science and experiment
software repositories.

## Rules

1. You are the coordinator, not the repository-file expert. You know indexed
   local repositories exist, but you do not know their contents in advance.
2. For every repository question, code-reading task, staged-input inspection,
   or request about how the local neutron-science or experiment software works,
   delegate first to the `code-chat-expert` subagent via the `task` tool.
3. When delegating repository investigation, explicitly ask the subagent to use
   indexed search first: use `search_code_hybrid` with `repo_id="all"` when
   the relevant repository is unknown, use `search_code_semantic` or
   `search_docs_local` when they better match the task, and use filesystem
   tools only to verify the few best indexed hits.
4. Keep the coordinator thread focused on orchestration: create a todo list for
   multi-step requests, delegate repository investigation, then synthesize the
   cited findings into the final answer.
5. Treat subagent returns as structured research packets. Compare packets,
   reconcile conflicts, remove duplication, and write the user-facing synthesis
   yourself.
6. Do not answer repository or codebase questions from model memory. Wait for
   cited evidence from the subagent, Context7, or Tavily before answering.
7. Local indexed repositories come first for coding questions. Context7 comes
   second for external library or framework documentation. Tavily web search is
   the last resort for any unresolved gap or explicitly web/current questions.
8. If the user explicitly says `use context7`, you must call an available
   Context7 tool before answering. If Context7 tools are unavailable, say that
   clearly.
9. If the user explicitly asks you to search the web, first establish whether
   the subagent and Context7 already cover the coding question, then use Tavily
   only for the remaining gap. If Tavily is unavailable, say that clearly.
10. Use Tavily only for facts that genuinely require web search. Keep Tavily
   queries focused and refine only if the first result set is insufficient.
11. Every substantive claim in the final answer must be grounded in cited local
   repository evidence, named external documentation from Context7, or linked
   Tavily web results. Never present unsupported model memory as fact.
12. Indexed search previews are leads, not evidence. If a subagent packet only
    cites search previews or lacks `read_file` line-number evidence for an
    important local claim, delegate a narrower follow-up or clearly mark the
    gap.
13. Your final response must be concise, directly answer the question, and cite
    file paths with line numbers whenever the subagent provides enough local
    evidence. Label Context7 claims as external documentation and Tavily claims
    as web search.
14. If the user's message is just a greeting, respond directly without
    delegation.
15. Do not expose internal scratch notes, todos, or subagent mechanics unless
    the user explicitly asks for them.
"""

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


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """Filesystem backend wrapper that blocks mutations under /repos."""

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        return WriteResult(error=f"Cannot write to read-only repository path '{file_path}'.")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return EditResult(error=f"Cannot edit read-only repository path '{file_path}'.")


class ReadOnlyStateBackend(StateBackend):
    """Ephemeral state backend wrapper that blocks mutations under /inputs."""

    def _is_read_only_path(self, file_path: str) -> bool:
        return file_path == "/inputs" or file_path.startswith("/inputs/")

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        if self._is_read_only_path(file_path):
            return WriteResult(error=f"Cannot write to read-only staged input path '{file_path}'.")
        return super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        if self._is_read_only_path(file_path):
            return EditResult(error=f"Cannot edit read-only staged input path '{file_path}'.")
        return super().edit(file_path, old_string, new_string, replace_all=replace_all)


def _build_code_chat_backend(
    repo_cache_root: Path,
) -> Callable[[ToolRuntime[Any]], CompositeBackend]:
    """Create a Deep Agents backend factory with scratch + read-only repos."""

    repos_backend = ReadOnlyFilesystemBackend(
        root_dir=repo_cache_root,
        virtual_mode=True,
    )

    def backend_factory(runtime: ToolRuntime[Any]) -> CompositeBackend:
        return CompositeBackend(
            default=ReadOnlyStateBackend(runtime),
            routes={
                "/repos/": repos_backend,
            },
        )

    return backend_factory


def _build_repo_research_subagent(
    llm_subagent: Any,
    llm_summarizer: Any,
    tools: list[Any],
    backend: Any,
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

async def create_code_chat_agent(
    provider: str,
    model: str,
) -> tuple[CodeChatAgentResources, CompiledStateGraph]:
    """Factory called by the agent registry on first use."""

    repo_manager = RepoManager()
    vector_index = RepoVectorIndex(repo_manager)
    code_chat_backend = _build_code_chat_backend(repo_manager.cache_dir)

    # Bootstrap is expected to happen before the server starts. The agent only
    # validates that the local repo cache and vector indices already exist.
    validate_bootstrap_ready(repo_manager, vector_index)

    llm = create_llm_with_fallback(provider=provider, model=model)
    llm_subagent = LLMFactory.create_llm(
        provider=CODE_CHAT_STATIC_PROVIDER,
        model=CODE_CHAT_SUBAGENT_MODEL,
        temperature=0.0,
    )
    llm_summarizer = LLMFactory.create_llm(
        provider=CODE_CHAT_STATIC_PROVIDER,
        model=CODE_CHAT_SUMMARIZER_MODEL,
        temperature=0.0,
    )
    local_tools = build_code_chat_tools(repo_manager, vector_index)
    context7_runtime = await load_optional_context7_tools()
    context7_tools = list(context7_runtime.tools) if context7_runtime is not None else []
    repo_subagent_tools = list(local_tools)
    repo_subagent_tools.extend(context7_tools)
    tavily_tool = load_optional_tavily_tool()
    coordinator_tools = list(context7_tools)
    if tavily_tool is not None:
        coordinator_tools.append(tavily_tool)

    repo_research_subagent = _build_repo_research_subagent(
        llm_subagent,
        llm_summarizer,
        repo_subagent_tools,
        code_chat_backend,
    )

    agent = create_agent(
        model=llm,
        tools=coordinator_tools,
        system_prompt=CODE_CHAT_SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SubAgentMiddleware(
                backend=code_chat_backend,
                subagents=[repo_research_subagent],
            ),
            create_summarization_middleware(llm_summarizer, code_chat_backend),
            PatchToolCallsMiddleware(),
            RuntimeModelMiddleware(),
        ],
        state_schema=FilesystemState,
        context_schema=RuntimeModelContext,
        checkpointer=get_checkpointer(),
        name="code_chat_agent",
    ).with_config({"recursion_limit": 1000})

    resources = CodeChatAgentResources(
        app=agent,
        context7_runtime=context7_runtime,
    )
    return (resources, agent)


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_agent_factory("code_chat_agent", create_code_chat_agent)
