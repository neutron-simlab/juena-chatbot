"""Tests for supervisor assembly and shared runtime resilience."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse, TodoListMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from juena.agents import juena_agent
from juena.indexing.bootstrap import RepositorySurvey
from juena.agents import specialist_runtime
from juena.agents.loop_guard import RepeatedToolCallMiddleware
from juena.agents.specialists.neutron_research.agent import (
    NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
    NEUTRON_RESEARCH_SPECIALIST_NAME,
)
from juena.agents.specialists.software.agent import (
    SOFTWARE_SPECIALIST_DESCRIPTION,
    SOFTWARE_SPECIALIST_NAME,
)
from juena.server.agent import registry as agent_registry
from juena.server.agent.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware


def test_fallback_models_do_not_depend_on_request_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, str]] = []
    sentinel = object()

    monkeypatch.setattr(specialist_runtime.global_config, "FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(specialist_runtime, "get_available_providers", lambda: {"openai": True})
    monkeypatch.setattr(specialist_runtime, "get_default_model", lambda _provider: "gpt-test")
    monkeypatch.setattr(
        specialist_runtime,
        "build_chat_model",
        lambda **kwargs: built.append(kwargs) or sentinel,
    )

    assert specialist_runtime.build_fallback_models() == [sentinel]
    assert built == [{"provider": "openai", "model": "gpt-test"}]


def test_fallback_models_are_skipped_when_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(specialist_runtime.global_config, "FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(specialist_runtime, "get_available_providers", lambda: {"openai": False})

    assert specialist_runtime.build_fallback_models() == []


def test_resilience_middleware_order_and_limits() -> None:
    middleware = specialist_runtime.resilience_middleware(
        [object()], model_call_limit=10, tool_call_limit=20
    )
    types = [type(item).__name__ for item in middleware]

    assert types.index("ModelFallbackMiddleware") < types.index("ModelRetryMiddleware")
    assert "ModelCallLimitMiddleware" in types
    assert "ToolCallLimitMiddleware" in types
    assert "ToolRetryMiddleware" in types
    retry = next(item for item in middleware if type(item).__name__ == "ModelRetryMiddleware")
    assert retry.max_retries == specialist_runtime.MODEL_MAX_RETRIES
    assert retry.on_failure == "error"


def test_model_fallback_runs_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = object()
    fallback = object()
    request = ModelRequest(model=primary, messages=[])
    middleware = specialist_runtime.resilience_middleware(
        [fallback], model_call_limit=10, tool_call_limit=20
    )
    fallback_middleware = next(
        item for item in middleware if type(item).__name__ == "ModelFallbackMiddleware"
    )
    retry_middleware = next(
        item for item in middleware if type(item).__name__ == "ModelRetryMiddleware"
    )
    attempts: list[object] = []

    monkeypatch.setattr("langchain.agents.middleware.model_retry.time.sleep", lambda _delay: None)

    def call_model(current_request: ModelRequest) -> ModelResponse:
        attempts.append(current_request.model)
        if current_request.model is primary:
            raise RuntimeError("primary unavailable")
        return ModelResponse(result=[AIMessage(content="fallback succeeded")])

    response = fallback_middleware.wrap_model_call(
        request,
        lambda current_request: retry_middleware.wrap_model_call(current_request, call_model),
    )

    assert attempts == [primary, primary, primary, primary, fallback]
    assert response.result[0].text == "fallback succeeded"


def test_resilience_middleware_omits_fallback_without_models() -> None:
    types = [
        type(item).__name__
        for item in specialist_runtime.resilience_middleware(
            [], model_call_limit=1, tool_call_limit=1
        )
    ]
    assert "ModelFallbackMiddleware" not in types


@pytest.mark.asyncio
async def test_supervisor_assembly_registers_both_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    fake_graph = object()
    supervisor_model = object()
    summarizer_model = object()
    context7_tool = object()
    tavily_tool = object()
    fallback_model = object()
    context7_runtime = SimpleNamespace(tools=[context7_tool])
    store = InMemoryStore()

    class StubRepoManager:
        def __init__(self) -> None:
            self.cache_dir = Path("/tmp/test-repos")

    class StubVectorIndex:
        """`build_search_index` takes no arguments: the URL is its only input."""

    software = {
        "name": SOFTWARE_SPECIALIST_NAME,
        "description": SOFTWARE_SPECIALIST_DESCRIPTION,
        "runnable": RunnableLambda(lambda state: state),
    }
    neutron = {
        "name": NEUTRON_RESEARCH_SPECIALIST_NAME,
        "description": NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
        "runnable": RunnableLambda(lambda state: state),
    }

    def fake_build_chat_model(**kwargs: Any) -> object:
        created.setdefault("model_calls", []).append(kwargs)
        return supervisor_model if kwargs["model"] == "user-model" else summarizer_model

    def fake_build_software_specialist(**kwargs: Any) -> dict[str, Any]:
        created["software_kwargs"] = kwargs
        return software

    def fake_build_neutron_specialist(**kwargs: Any) -> dict[str, Any]:
        created["neutron_kwargs"] = kwargs
        return neutron

    def fake_create_agent(**kwargs: Any) -> object:
        created["agent_kwargs"] = kwargs

        class FakeGraph:
            def with_config(self, config: dict[str, Any]) -> object:
                created["graph_config"] = config
                return fake_graph

        return FakeGraph()

    async def fake_context7_loader() -> Any:
        return context7_runtime

    monkeypatch.setattr(juena_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(juena_agent, "build_search_index", StubVectorIndex)
    monkeypatch.setattr(
        juena_agent,
        "survey_repositories",
        lambda *_args: RepositorySurvey(searchable=["jscatter"], problems=[]),
    )
    monkeypatch.setattr(juena_agent, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(juena_agent, "load_optional_context7_tools", fake_context7_loader)
    monkeypatch.setattr(juena_agent, "load_optional_tavily_tool", lambda: tavily_tool)
    monkeypatch.setattr(juena_agent, "build_fallback_models", lambda: [fallback_model])
    monkeypatch.setattr(juena_agent, "build_software_specialist", fake_build_software_specialist)
    monkeypatch.setattr(
        juena_agent,
        "build_neutron_research_specialist",
        fake_build_neutron_specialist,
    )
    monkeypatch.setattr(juena_agent, "create_summarization_middleware", lambda *_args: object())
    monkeypatch.setattr(juena_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(juena_agent, "get_checkpointer", lambda: object())
    monkeypatch.setattr(juena_agent, "get_store", lambda: store)

    resources, graph = await juena_agent.create_juena_agent("openai", "user-model")

    assert graph is fake_graph
    assert isinstance(resources, juena_agent.JuenaAgentResources)
    assert resources.context7_runtime is context7_runtime
    assert created["model_calls"] == [
        {"provider": "openai", "model": "user-model"},
        {
            "provider": juena_agent.SUMMARIZER_PROVIDER,
            "model": juena_agent.SUMMARIZER_MODEL,
            "temperature": 0.0,
        },
    ]
    assert created["software_kwargs"]["has_repo_research"] is True
    assert created["software_kwargs"]["context7_tools"] == (context7_tool,)
    assert created["neutron_kwargs"]["tavily_tool"] is tavily_tool
    assert created["software_kwargs"]["fallback_models"] == [fallback_model]
    assert created["neutron_kwargs"]["fallback_models"] == [fallback_model]

    agent_kwargs = created["agent_kwargs"]
    assert agent_kwargs["model"] is supervisor_model
    assert [tool.name for tool in agent_kwargs["tools"]] == ["ask_user"]
    assert agent_kwargs["context_schema"] is RuntimeModelContext
    assert agent_kwargs["store"] is store
    assert created["graph_config"] == {"recursion_limit": 1000}

    middleware = agent_kwargs["middleware"]
    assert any(isinstance(item, MemoryMiddleware) for item in middleware)
    assert any(isinstance(item, FilesystemMiddleware) for item in middleware)
    assert any(isinstance(item, RuntimeModelMiddleware) for item in middleware)
    assert not any(isinstance(item, SkillsMiddleware) for item in middleware)

    assert any(isinstance(item, RepeatedToolCallMiddleware) for item in middleware)
    assert not any(isinstance(item, TodoListMiddleware) for item in middleware)

    subagents = next(item for item in middleware if isinstance(item, SubAgentMiddleware))
    assert subagents.subagent_names == {
        SOFTWARE_SPECIALIST_NAME,
        NEUTRON_RESEARCH_SPECIALIST_NAME,
    }
    task_instructions = subagents.tools[0].description
    assert SOFTWARE_SPECIALIST_NAME in task_instructions
    assert SOFTWARE_SPECIALIST_DESCRIPTION in task_instructions
    assert NEUTRON_RESEARCH_SPECIALIST_NAME in task_instructions
    assert NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION in task_instructions
    assert "<verified_by_server>" in task_instructions
    assert "Launch multiple agents concurrently whenever possible" not in task_instructions


@pytest.mark.asyncio
async def test_missing_optional_capabilities_do_not_remove_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    class StubRepoManager:
        cache_dir = Path("/tmp/unavailable-repos")

    class StubVectorIndex:
        pass

    async def no_context7() -> None:
        return None

    def specialist(name: str, description: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "runnable": RunnableLambda(lambda state: state),
        }

    def build_software(**kwargs: Any) -> dict[str, Any]:
        calls["software"] = kwargs
        return specialist(SOFTWARE_SPECIALIST_NAME, SOFTWARE_SPECIALIST_DESCRIPTION)

    def build_neutron(**kwargs: Any) -> dict[str, Any]:
        calls["neutron"] = kwargs
        return specialist(
            NEUTRON_RESEARCH_SPECIALIST_NAME,
            NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
        )

    class FakeGraph:
        def with_config(self, _config: dict[str, Any]) -> "FakeGraph":
            return self

    monkeypatch.setattr(juena_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(juena_agent, "build_search_index", StubVectorIndex)
    monkeypatch.setattr(
        juena_agent,
        "survey_repositories",
        lambda *_args: RepositorySurvey(searchable=[], problems=["missing"]),
    )
    monkeypatch.setattr(juena_agent, "build_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(juena_agent, "load_optional_context7_tools", no_context7)
    monkeypatch.setattr(juena_agent, "load_optional_tavily_tool", lambda: None)
    monkeypatch.setattr(juena_agent, "build_fallback_models", lambda: [])
    monkeypatch.setattr(juena_agent, "build_software_specialist", build_software)
    monkeypatch.setattr(juena_agent, "build_neutron_research_specialist", build_neutron)
    monkeypatch.setattr(juena_agent, "create_summarization_middleware", lambda *_args: object())
    monkeypatch.setattr(juena_agent, "create_agent", lambda **_kwargs: FakeGraph())
    monkeypatch.setattr(juena_agent, "get_checkpointer", lambda: object())
    monkeypatch.setattr(juena_agent, "get_store", InMemoryStore)

    await juena_agent.create_juena_agent("openai", "user-model")

    assert calls["software"]["has_repo_research"] is False
    assert calls["software"]["context7_tools"] == ()
    assert calls["neutron"]["tavily_tool"] is None


def test_only_the_juena_agent_is_registered() -> None:
    assert agent_registry.list_registered_agents() == [juena_agent.JUENA_AGENT_ID]


@pytest.mark.asyncio
async def test_shutdown_agents_closes_cached_context7_clients() -> None:
    closed: list[bool] = []

    class FakeClient:
        async def aclose(self) -> None:
            closed.append(True)

    existing = dict(agent_registry._agent_registry)
    agent_registry._agent_registry.clear()
    agent_registry._agent_registry["test-agent"] = (
        SimpleNamespace(context7_runtime=SimpleNamespace(client=FakeClient())),
        object(),
    )

    try:
        await agent_registry.shutdown_agents()
    finally:
        agent_registry._agent_registry.clear()
        agent_registry._agent_registry.update(existing)

    assert closed == [True]
