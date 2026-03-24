"""
Code-Chat Agent – answers questions about software repositories using
hybrid (keyword + embedding) search over local code and documentation.

Registration happens at import time via ``register_agent_factory``.
"""

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
3. Repository file navigation should use Deep Agents filesystem primitives:
   `ls`, `glob`, `grep`, and `read_file` against `/repos/...`.
4. Use the custom semantic retrieval tools only when conceptual or indexed
   retrieval is more useful than direct filesystem navigation.
5. Never answer codebase questions from memory alone. Wait for the subagent's
   cited findings before you synthesize the final answer.
6. Your final response to the user must be concise, directly answer the
   question, and cite file paths with line numbers whenever the subagent
   provides enough evidence.
7. If the user's message is just a greeting or is outside repository analysis,
   respond directly without delegation.
8. Do not expose internal scratch notes, todos, or subagent mechanics unless
   the user explicitly asks for them.
"""

CODE_CHAT_RESEARCH_SUBAGENT_PROMPT = """\
You are the repository-research specialist for indexed software repositories.

## Required workflow

1. Always search first. Do not answer from memory.
2. Discover available repositories by listing `/repos`.
3. For exact repository navigation, use filesystem tools on `/repos/...`:
   - `ls("/repos")` to discover repository ids.
   - `glob` and `grep` for exact filenames, symbols, and strings.
   - `read_file` for line-numbered source and documentation evidence.
4. Use the custom retrieval tools when indexed search is the better fit:
   - `search_code_hybrid` for most technical questions.
   - `search_code_semantic` for conceptual questions.
   - `search_docs_local` for README and docs questions.
5. Search results include a canonical `/repos/...` path. When a result looks
   relevant, read that path with `read_file` for exact evidence.
6. Never write to or edit files under `/repos`. Those paths are read-only.
7. Keep any scratch files short and disposable. Use them only to compress your
   own working notes, not as a substitute for evidence.

## Output requirements

- Return a compact report to the coordinator, not a user-facing essay.
- Include the best answer first.
- Cite the supporting file path and line numbers for each important claim.
- If the evidence is ambiguous or incomplete, say so clearly.
"""

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
            routes={"/repos/": repos_backend},
        )

    return backend_factory


def _build_repo_research_subagent(
    llm: Any,
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    backend: Any,
) -> CompiledSubAgent:
    """Create the repo-analysis subagent with bound retrieval tools."""

    tools = build_code_chat_tools(repo_manager, vector_index)

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
            "and multi-step questions about indexed repositories."
        ),
        "runnable": subagent,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_code_chat_agent(
    provider: str,
    model: str,
) -> tuple[CompiledStateGraph, CompiledStateGraph]:
    """Factory called by the agent registry on first use."""

    repo_manager = RepoManager()
    vector_index = RepoVectorIndex(repo_manager)
    backend = _build_code_chat_backend(repo_manager.cache_dir)

    # Bootstrap is expected to happen before the server starts. The agent only
    # validates that the local repo cache and vector indices already exist.
    validate_bootstrap_ready(repo_manager, vector_index)

    llm = create_llm_with_fallback(provider=provider, model=model)
    repo_research_subagent = _build_repo_research_subagent(
        llm,
        repo_manager,
        vector_index,
        backend,
    )

    agent = create_deep_agent(
        model=llm,
        tools=[],
        system_prompt=CODE_CHAT_SYSTEM_PROMPT,
        subagents=[repo_research_subagent],
        backend=backend,
        checkpointer=get_checkpointer(),
        name="code_chat_agent",
    )

    return (agent, agent)


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_agent_factory("code_chat_agent", create_code_chat_agent)
