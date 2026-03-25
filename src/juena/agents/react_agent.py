"""
General-purpose chat agent with optional Tavily web search.
"""

from typing import Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from juena.core.llms_providers import create_llm_with_fallback
from juena.server.agent_registry import register_agent_factory
from juena.server.checkpointer import get_checkpointer
from juena.tools.tavily import load_optional_tavily_tool

REACT_SYSTEM_PROMPT = """\
You are a helpful general-purpose assistant.

## Tool-use rules

1. Use Tavily web search for:
   - current or recent events
   - live facts that can change over time
   - external factual questions that are not already grounded in the current chat
   - external factual questions when you are uncertain
   - explicit requests to search the web
2. Prefer Tavily-grounded answers over unverified model memory for factual
   claims about the outside world.
3. If Tavily is unavailable, say you cannot verify with web search right now.
4. Start with one precise search query. Only search again if the first search
   is insufficient.
5. Do not run broad or repetitive search loops.
6. Keep web-search answers concise and include source links when Tavily was
   used.
7. If the answer does not need web verification, respond directly without
   using the tool.

Always be clear, concise, and helpful.
"""


async def create_react_agent(
    provider: str,
    model: str,
) -> tuple[Any, CompiledStateGraph]:
    """Create the general-purpose agent."""

    llm = create_llm_with_fallback(provider=provider, model=model)
    tavily_tool = load_optional_tavily_tool()
    tools = [tavily_tool] if tavily_tool is not None else []

    react_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=REACT_SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
        name="react_agent",
    )

    return (None, react_agent)


register_agent_factory("react_agent", create_react_agent, set_as_default=True)
