"""Tests for react_agent Tavily wiring."""

from __future__ import annotations

from typing import Any

import pytest

from juena.agents import react_agent
from juena.server.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware


@pytest.mark.asyncio
async def test_create_react_agent_uses_tavily_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    llm = object()
    tavily_tool = object()
    fake_graph = object()

    monkeypatch.setattr(react_agent, "create_llm_with_fallback", lambda **kwargs: llm)
    monkeypatch.setattr(react_agent, "load_optional_tavily_tool", lambda: tavily_tool)
    monkeypatch.setattr(react_agent, "get_checkpointer", lambda: object())

    def fake_create_agent(**kwargs: Any) -> object:
        created["kwargs"] = kwargs
        return fake_graph

    monkeypatch.setattr(react_agent, "create_agent", fake_create_agent)

    resources, graph = await react_agent.create_react_agent("openai", "gpt-test")

    assert resources is None
    assert graph is fake_graph
    assert created["kwargs"]["model"] is llm
    assert created["kwargs"]["tools"] == [tavily_tool]
    assert created["kwargs"]["context_schema"] is RuntimeModelContext
    assert len(created["kwargs"]["middleware"]) == 1
    assert isinstance(created["kwargs"]["middleware"][0], RuntimeModelMiddleware)
    assert "Tavily" in created["kwargs"]["system_prompt"]
    assert "one precise search query" in created["kwargs"]["system_prompt"]
    assert "Prefer Tavily-grounded answers over unverified model memory" in created["kwargs"]["system_prompt"]
    assert "include source links" in created["kwargs"]["system_prompt"]
    assert not hasattr(react_agent, "get_weather")


@pytest.mark.asyncio
async def test_create_react_agent_without_tavily_still_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    llm = object()
    fake_graph = object()

    monkeypatch.setattr(react_agent, "create_llm_with_fallback", lambda **kwargs: llm)
    monkeypatch.setattr(react_agent, "load_optional_tavily_tool", lambda: None)
    monkeypatch.setattr(react_agent, "get_checkpointer", lambda: object())

    def fake_create_agent(**kwargs: Any) -> object:
        created["kwargs"] = kwargs
        return fake_graph

    monkeypatch.setattr(react_agent, "create_agent", fake_create_agent)

    _, graph = await react_agent.create_react_agent("openai", "gpt-test")

    assert graph is fake_graph
    assert created["kwargs"]["tools"] == []
    assert created["kwargs"]["context_schema"] is RuntimeModelContext
    assert isinstance(created["kwargs"]["middleware"][0], RuntimeModelMiddleware)
