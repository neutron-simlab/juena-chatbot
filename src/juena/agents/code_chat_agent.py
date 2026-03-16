"""
Code-Chat Agent – answers questions about software repositories using
hybrid (keyword + embedding) search over local code and documentation.

Registration happens at import time via ``register_agent_factory``.
"""

from typing import Any, List, Tuple

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict

from juena.core.llms_providers import create_llm_with_fallback
from juena.server.agent_registry import register_agent_factory
from juena.server.checkpointer import get_checkpointer
from juena.tools.repo_search import CODE_CHAT_TOOLS, ensure_indices_built

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODE_CHAT_SYSTEM_PROMPT = """\
You are a knowledgeable code assistant that helps users understand software \
repositories.  You have access to one or more indexed codebases and their \
documentation.

## Rules

1. **Always start by listing repositories** (`list_repositories`) so you know \
which `repo_id` values are available.
2. **For every technical question, search FIRST** – never answer from memory alone.
   - Use `search_code_hybrid` (recommended) for most queries – it combines \
keyword and semantic search.
   - Use `search_code_keyword` when the user asks for an exact symbol, string, \
or function name.
   - Use `search_code_semantic` for conceptual / "how does X work?" questions.
   - Use `search_docs_local` to search README / docs specifically.
3. **Read deeper context** with `read_repo_file` when a search hit looks \
relevant but the snippet is too short.
4. When answering:
   - Always cite the file path and line number(s) so the user can verify.
   - If relevant code spans multiple files, show the key pieces.
   - Explain *why* the code works the way it does, not just *what* it does.
5. If the user's question is a greeting or off-topic, reply politely and \
remind them of the repos you can help with.

## Available tools

- `list_repositories` – discover available repos
- `search_code_hybrid` – hybrid keyword + semantic search (preferred)
- `search_code_keyword` – exact-string grep search
- `search_code_semantic` – embedding similarity search
- `search_docs_local` – search docs-only chunks
- `read_repo_file` – read a file or line range from a repo
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CodeChatState(TypedDict):
    messages: List[BaseMessage]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_code_chat_agent(
    provider: str,
    model: str,
) -> Tuple[Any, CompiledStateGraph]:
    """Factory called by the agent registry on first use."""

    # Ensure all repo vector indices are built before the agent can be used
    ensure_indices_built()

    llm = create_llm_with_fallback(provider=provider, model=model)

    agent = create_agent(
        model=llm,
        tools=CODE_CHAT_TOOLS,
        system_prompt=CODE_CHAT_SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
        name="code_chat_agent",
    )

    return (None, agent)


# ---------------------------------------------------------------------------
# Self-register
# ---------------------------------------------------------------------------

register_agent_factory("code_chat_agent", create_code_chat_agent)
