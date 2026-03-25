"""
Code-Chat Agent – answers questions about software repositories using
hybrid (keyword + embedding) search over local code and documentation.

Registration happens at import time via ``register_agent_factory``.
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import ToolRuntime
from langgraph.graph.state import CompiledStateGraph

from juena.core.llms_providers import create_llm_with_fallback
from juena.retrieval.bootstrap import validate_bootstrap_ready
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex
from juena.server.agent_registry import register_agent_factory
from juena.server.checkpointer import get_checkpointer
from juena.tools.context7 import Context7Runtime, load_optional_context7_tools
from juena.tools.repo_search import (
    build_code_chat_tools,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CODE_CHAT_SYSTEM_PROMPT = """\
You are a code-chat coordinator for indexed software repositories.

## Rules

1. For technical repository questions, delegate the investigation to the
   `general-purpose` subagent via the `task` tool. Keep repository search and
   file reading out of the coordinator thread whenever possible.
2. Configured repositories are mounted read-only at `/repos/<repo_id>/`.
   If the user asks which repositories are available, inspect `/repos`.
3. If `/inputs/` exists, inspect it before searching repositories. Treat
   `/inputs/...` as turn-scoped user-provided code, logs, configs, or docs.
4. Repository file navigation should use Deep Agents filesystem primitives:
   `ls`, `glob`, `grep`, and `read_file` against `/repos/...`.
5. Use paginated `read_file` calls for large staged input files instead of
   trying to ingest them all at once.
6. Use the local repository retrieval tools for configured repositories when
   conceptual or indexed retrieval is more useful than direct filesystem
   navigation.
7. Use Context7 tools only for external libraries, frameworks, and dependency
   documentation. They complement local repo analysis and do not replace
   `/repos/...` evidence.
8. If the user explicitly says `use context7`, you must call an available
   Context7 tool before answering. If Context7 tools are unavailable, say that
   clearly.
9. The repository research subagent may use Context7 to enrich its findings
   when local repository answers depend on upstream libraries, frameworks, or
   dependency documentation.
10. Never answer codebase questions from memory alone. Wait for the subagent's
   cited findings before you synthesize the final answer.
11. Your final response to the user must be concise, directly answer the
   question, and cite file paths with line numbers whenever the subagent
   provides enough local repository evidence. If you use external Context7
   material, label it as external documentation rather than local repository
   evidence.
12. If the user's message is just a greeting or is outside repository analysis,
   respond directly without delegation.
13. Do not expose internal scratch notes, todos, or subagent mechanics unless
   the user explicitly asks for them.
14. Treat `/inputs/...` as read-only user evidence. Never write the user's
   solution into staged files. If the user asks for a fix, provide the
   proposed code changes directly in your chat response.
"""

CODE_CHAT_RESEARCH_SUBAGENT_PROMPT = """\
You are the repository-research specialist for indexed software repositories.

## Required workflow

1. Always search first. Do not answer from memory.
2. If `/inputs/` exists, inspect it before searching repositories. Treat it as
   user-provided evidence for the current turn.
3. Discover available repositories by listing `/repos`.
4. For exact repository navigation, use filesystem tools on `/repos/...`:
   - `ls("/repos")` to discover repository ids.
   - `glob` and `grep` for exact filenames, symbols, and strings.
   - `read_file` for line-numbered source and documentation evidence.
5. For large staged user files, page through `/inputs/...` with `read_file`
   offsets and use `grep` before reading more.
6. Use the local retrieval tools when indexed search is the better fit for a
   configured repository:
   - `search_code_hybrid` for most technical questions.
   - `search_code_semantic` for conceptual questions.
   - `search_docs_local` for README and docs questions.
7. Use Context7 tools only for upstream libraries, frameworks, and dependency
   docs/examples. Resolve the library first, then fetch the relevant docs.
8. If the user explicitly says `use context7`, you must call an available
   Context7 tool before answering. If no Context7 tool is available, say that
   clearly.
9. Use Context7 to enrich local repository findings when the answer depends on
   upstream framework behavior, external APIs, or dependency docs that are not
   fully explained in the configured repository itself.
10. Search results from local repo tools include a canonical `/repos/...` path.
   When a result looks
   relevant, read that path with `read_file` for exact evidence.
11. Prefer local repository evidence over Context7 whenever the answer depends on
   configured repository behavior or implementation details.
12. Never invent local file paths or line numbers for Context7 results. Cite
   Context7 findings as external documentation and name the library when you use
   them.
13. Never write to or edit files under `/repos`. Those paths are read-only.
14. Ask a concise follow-up only when the staged user inputs and repository
    evidence are still insufficient for a useful answer.
15. Keep any scratch files short and disposable. Use them only to compress your
   own working notes, not as a substitute for evidence.
16. Never write to or edit files under `/inputs`. Those paths are read-only
   staged user evidence. When suggesting a fix, return the changed code or
   patch in the response instead of editing staged files.
17. When the answer depends on external library behavior and the relevant files
   are read-only, prefer using available Context7 tools to ground the
   recommendation before you answer.

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

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        return WriteResult(error=f"Cannot write to read-only staged input path '{file_path}'.")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return EditResult(error=f"Cannot edit read-only staged input path '{file_path}'.")


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
            default=StateBackend(runtime),
            routes={
                "/inputs/": ReadOnlyStateBackend(runtime),
                "/repos/": repos_backend,
            },
        )

    return backend_factory


def _build_repo_research_subagent(
    llm: Any,
    tools: list[Any],
    backend: Any,
) -> CompiledSubAgent:
    """Create the repo-analysis subagent with bound retrieval tools."""

    subagent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=CODE_CHAT_RESEARCH_SUBAGENT_PROMPT,
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(backend=backend),
            create_summarization_middleware(llm, backend),
            PatchToolCallsMiddleware(),
        ],
        name="code_chat_repo_research_subagent",
    )
    return {
        "name": "general-purpose",
        "description": (
            "Use this agent for repository analysis, code search, file reading, "
            "multi-step questions about indexed repositories, and enriching "
            "findings with Context7 documentation for external dependencies "
            "and upstream frameworks."
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
    backend = _build_code_chat_backend(repo_manager.cache_dir)

    # Bootstrap is expected to happen before the server starts. The agent only
    # validates that the local repo cache and vector indices already exist.
    validate_bootstrap_ready(repo_manager, vector_index)

    llm = create_llm_with_fallback(provider=provider, model=model)
    local_tools = build_code_chat_tools(repo_manager, vector_index)
    context7_runtime = await load_optional_context7_tools()
    context7_tools = list(context7_runtime.tools) if context7_runtime is not None else []
    combined_tools = list(local_tools)
    combined_tools.extend(context7_tools)

    repo_research_subagent = _build_repo_research_subagent(
        llm,
        combined_tools,
        backend,
    )

    agent = create_deep_agent(
        model=llm,
        tools=context7_tools,
        system_prompt=CODE_CHAT_SYSTEM_PROMPT,
        subagents=[repo_research_subagent],
        backend=backend,
        checkpointer=get_checkpointer(),
        name="code_chat_agent",
    )

    resources = CodeChatAgentResources(
        app=agent,
        context7_runtime=context7_runtime,
    )
    return (resources, agent)


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_agent_factory("code_chat_agent", create_code_chat_agent)
