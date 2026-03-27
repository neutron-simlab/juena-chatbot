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
from deepagents.middleware.filesystem import FilesystemMiddleware
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
from juena.indexing.sparse_index import RepoSparseIndex
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
3. Keep the coordinator thread focused on orchestration: create a todo list for
   multi-step requests, delegate repository investigation, then synthesize the
   cited findings into the final answer.
4. Do not answer repository or codebase questions from model memory. Wait for
   cited evidence from the subagent, Context7, or Tavily before answering.
5. Local indexed repositories come first for coding questions. Context7 comes
   second for external library or framework documentation. Tavily web search is
   the last resort for any unresolved gap or explicitly web/current questions.
6. If the user explicitly says `use context7`, you must call an available
   Context7 tool before answering. If Context7 tools are unavailable, say that
   clearly.
7. If the user explicitly asks you to search the web, first establish whether
   the subagent and Context7 already cover the coding question, then use Tavily
   only for the remaining gap. If Tavily is unavailable, say that clearly.
8. Use Tavily only for facts that genuinely require web search. Keep Tavily
   queries focused and refine only if the first result set is insufficient.
9. Every substantive claim in the final answer must be grounded in cited local
   repository evidence, named external documentation from Context7, or linked
   Tavily web results. Never present unsupported model memory as fact.
10. Your final response must be concise, directly answer the question, and cite
    file paths with line numbers whenever the subagent provides enough local
    evidence. Label Context7 claims as external documentation and Tavily claims
    as web search.
11. If the user's message is just a greeting, respond directly without
    delegation.
12. Do not expose internal scratch notes, todos, or subagent mechanics unless
    the user explicitly asks for them.
"""

CODE_CHAT_RESEARCH_SUBAGENT_PROMPT = """\
You are the repository expert for indexed neutron-science and experiment
software repositories.

## Required workflow

1. Always search first. Do not answer from memory.
2. Start repository investigation with `search_code_hybrid` unless the user is
   only asking which repositories exist. `search_code_hybrid` is the primary
   tool because it combines FTS5 keyword retrieval with vector-semantic search
   over the indexed repositories for the best recall.
3. After relevant hybrid hits appear, use filesystem tools on `/repos/...` to
   validate the result with exact evidence from the real files.
4. If `/inputs/` exists, inspect it before or alongside repository search when
   the user's question depends on staged files. Treat
   `/inputs/uploads/...` as persistent chat uploads and `/inputs/current_*` as
   turn-scoped helper files for the latest message.
5. If present, inspect `/inputs/uploads_manifest.md` to discover the full set
   of uploads available in this chat.
6. Discover available repositories by listing `/repos` when the user asks what
   is available or when you need repository ids for a search.
7. For exact repository navigation, use filesystem tools on `/repos/...`:
   - `ls("/repos")` to discover repository ids.
   - `glob` and `grep` for exact filenames, symbols, and strings.
   - `read_file` for line-numbered source and documentation evidence.
8. For large staged user files, page through `/inputs/...` with `read_file`
   offsets and use `grep` before reading more.
9. Use `search_code_semantic` only for narrow conceptual follow-ups when the
   hybrid results were already too broad, and use `search_docs_local` for
   README or documentation-only questions.
10. Exhaust local repository evidence before consulting external sources.
11. Use Context7 tools next for upstream libraries, frameworks, and dependency
   docs/examples. Resolve the library first, then fetch the relevant docs.
12. Tavily web search is not available in this subagent. Stay with local
   repository tools and Context7.
13. If the user explicitly says `use context7`, you must call an available
    Context7 tool before answering. If no Context7 tool is available, say that
    clearly.
14. Use Context7 to enrich local repository findings when the answer depends on
    upstream framework behavior, external APIs, or dependency docs that are not
    fully explained in the configured repository itself.
15. Search results from local repo tools include a canonical `/repos/...` path.
    When a result looks relevant, read that path with `read_file` for exact
    evidence.
16. Prefer local repository evidence over Context7 whenever the answer depends
    on configured repository behavior or implementation details.
17. Never invent local file paths or line numbers for Context7 results. Cite
    Context7 findings as external documentation and name the library when you
    use them.
18. Do not broaden the investigation with repetitive searches. Use one focused
    repo or Context7 lookup at a time and only refine when needed.
19. Never write to or edit files under `/repos`. Those paths are read-only.
20. Ask a concise follow-up only when the staged user inputs and repository
    evidence are still insufficient for a useful answer.
21. Keep any scratch files short and disposable. Use them only to compress your
    own working notes, not as a substitute for evidence.
22. Never write to or edit files under `/inputs`. Those paths are read-only
    staged user evidence. When suggesting a fix, return the changed code or
    patch in the response instead of editing staged files.
23. When the answer depends on external library behavior and the relevant files
    are read-only, prefer using available Context7 tools to ground the
    recommendation before you answer.
24. If the remaining gap is outside local repositories and outside Context7's
    scope, say so clearly so the coordinator can use Tavily web search as a
    last resort.

## Output requirements

- Return a compact report to the coordinator, not a user-facing essay.
- Include the best answer first.
- Cite the supporting file path and line numbers for each important local claim.
- For Context7-based claims, say they come from external documentation and name
  the library or framework you used.
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
            FilesystemMiddleware(backend=backend),
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
            "repositories. Start with `search_code_hybrid`, then validate with "
            "filesystem navigation and `read_file`. Use Context7 only for "
            "external dependencies or upstream frameworks. This subagent does "
            "not use Tavily web search."
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
    sparse_index = RepoSparseIndex(repo_manager)
    repo_backend = _build_code_chat_backend(repo_manager.cache_dir)

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
    local_tools = build_code_chat_tools(repo_manager, vector_index, sparse_index)
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
        repo_backend,
    )

    agent = create_agent(
        model=llm,
        tools=coordinator_tools,
        system_prompt=CODE_CHAT_SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SubAgentMiddleware(
                backend=ReadOnlyStateBackend,
                subagents=[repo_research_subagent],
            ),
            create_summarization_middleware(llm_summarizer, ReadOnlyStateBackend),
            PatchToolCallsMiddleware(),
            RuntimeModelMiddleware(),
        ],
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
