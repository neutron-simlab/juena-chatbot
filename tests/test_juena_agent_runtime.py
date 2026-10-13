"""Tests for supervisor assembly and shared runtime resilience."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import inspect

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
from juena_core.agents import specialist_runtime
import juena_core
from juena_core.agents.loop_guard import RepeatedToolCallMiddleware
from juena.agents.specialists.neutron_research.agent import (
    NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
    NEUTRON_RESEARCH_SPECIALIST_NAME,
)
from juena.agents.specialists.software.agent import (
    SOFTWARE_SPECIALIST_DESCRIPTION,
    SOFTWARE_SPECIALIST_NAME,
)
from juena_core.server.agent import registry as agent_registry
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware


def _override_fallback_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """Point core's `settings().FALLBACK_PROVIDER` at *provider* for one test.

    `build_fallback_models` reads core settings rather than this application's
    `global_config`, and `CoreSettings` is frozen and configured once per
    process, so the seam to patch is the accessor the module actually calls.
    """

    import dataclasses

    patched = dataclasses.replace(juena_core.settings(), FALLBACK_PROVIDER=provider)
    monkeypatch.setattr(specialist_runtime, "settings", lambda: patched)


def test_fallback_models_do_not_depend_on_request_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, str]] = []
    sentinel = object()

    _override_fallback_provider(monkeypatch, "openai")
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
    _override_fallback_provider(monkeypatch, "openai")
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


def _limiters(**kwargs: Any) -> list[Any]:
    return [
        item
        for item in specialist_runtime.resilience_middleware(
            [], model_call_limit=10, tool_call_limit=20, **kwargs
        )
        if type(item).__name__ in ("ModelCallLimitMiddleware", "ToolCallLimitMiddleware")
    ]


def test_only_the_model_call_limit_terminates() -> None:
    """`exit_behavior="end"` on a tool limiter crashes on a parallel batch.

    LangChain raises `NotImplementedError` -- "Cannot end execution with other
    tool calls pending" -- when the response that crossed the budget also carries
    another tool call. So the tool limiters stay a mechanical cap: the
    over-budget call does not run. The repeated-call guard closes the observed
    loop and the model-call limit is the final bound.
    """
    behaviours = {
        type(item).__name__: item.exit_behavior for item in _limiters()
    }

    assert behaviours["ModelCallLimitMiddleware"] == "end"
    assert all(
        item.exit_behavior == "continue"
        for item in _limiters()
        if type(item).__name__ == "ToolCallLimitMiddleware"
    )


def test_no_budget_spans_more_than_one_invocation() -> None:
    """Deliberately absent, not overlooked.

    A resume after an approval or an `ask_user` answer is a new invocation, so
    `run_limit` restarts -- but the stock `thread_limit` counts every later turn
    of the same chat too, which would eventually stop a healthy long
    conversation. Bounding a user *turn* needs a turn identity JueNA does not
    have. Until then the honest contract is per invocation.
    """
    assert all(item.thread_limit is None for item in _limiters())

    source = inspect.getsource(specialist_runtime.resilience_middleware)
    assert "thread_limit" not in source.split('"""')[2]


@pytest.mark.asyncio
async def test_a_long_conversation_is_not_ended_by_earlier_turns() -> None:
    """Driven through a graph, because the config assertion alone is not proof.

    Two turns on one `thread_id` whose combined model calls exceed the limit. If
    a `thread_limit` is ever added, the second turn stops answering and this
    fails -- which is the whole point of writing it down.
    """
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    agent = create_agent(
        model=ToolModel(responses=[AIMessage(content=f"answer {i}") for i in range(6)]),
        tools=[],
        middleware=specialist_runtime.resilience_middleware(
            [], model_call_limit=2, tool_call_limit=4
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "one-long-chat"}}

    answers = [
        (await agent.ainvoke({"messages": [("user", f"question {turn}")]}, config=config))[
            "messages"
        ][-1].text
        for turn in range(3)
    ]

    assert all("answer" in text for text in answers), (
        f"a later turn stopped answering: {answers}"
    )


def test_the_tool_retry_policy_is_pinned_not_inherited() -> None:
    """`ToolRetryMiddleware` defaults to 2 today; a silent upstream change to
    that default would silently change JueNA's policy."""

    retry = next(
        item
        for item in specialist_runtime.resilience_middleware(
            [], model_call_limit=10, tool_call_limit=20
        )
        if type(item).__name__ == "ToolRetryMiddleware"
    )

    assert retry.max_retries == specialist_runtime.TOOL_MAX_RETRIES == 2


def test_a_budget_on_an_absent_tool_is_not_built() -> None:
    """`ask_user` is not bound to a background specialist, so nothing can call it."""

    named = [
        item.tool_name for item in _limiters(ask_user_bound=False) if hasattr(item, "tool_name")
    ]

    assert "ask_user" not in named
    assert "ask_user" in [
        item.tool_name for item in _limiters() if hasattr(item, "tool_name")
    ]


@pytest.mark.asyncio
async def test_supervisor_assembly_registers_both_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    fake_graph = object()
    supervisor_model = object()
    summarizer_model = object()
    specialist_summarizer_model = object()
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
        if kwargs["model"].startswith("90 - Qwen 3.8-Flash-Next"):
            return specialist_summarizer_model
        return supervisor_model if kwargs["model"] == "user-model" else summarizer_model

    def fake_build_software_specialist(**kwargs: Any) -> dict[str, Any]:
        # Every call, not the last one: each specialist is now compiled twice,
        # once for `task` and once for background research.
        created.setdefault("software_calls", []).append(kwargs)
        created["software_kwargs"] = kwargs
        return software

    def fake_build_neutron_specialist(**kwargs: Any) -> dict[str, Any]:
        created.setdefault("neutron_calls", []).append(kwargs)
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
    monkeypatch.setattr(
        specialist_runtime, "create_summarization_middleware", lambda *_args: object()
    )
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
        {
            "provider": "blablador",
            "model": (
                "90 - Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster - "
                "An excellent model from September 2026"
            ),
            "temperature": 0.0,
            "max_tokens": 32_768,
        },
    ]
    assert created["software_kwargs"]["summarizer_model"] is specialist_summarizer_model
    assert created["neutron_kwargs"]["summarizer_model"] is specialist_summarizer_model
    assert created["software_kwargs"]["has_repo_research"] is True
    assert created["software_kwargs"]["context7_tools"] == (context7_tool,)
    assert created["neutron_kwargs"]["tavily_tool"] is tavily_tool
    assert created["software_kwargs"]["fallback_models"] == [fallback_model]
    assert created["neutron_kwargs"]["fallback_models"] == [fallback_model]

    # Background graphs are built only when a job owns its checkpoint connection.
    for calls in (created["software_calls"], created["neutron_calls"]):
        assert len(calls) == 1
        assert not calls[0].get("unattended", False)
    assert set(resources.background_specialists) == {
        SOFTWARE_SPECIALIST_NAME,
        NEUTRON_RESEARCH_SPECIALIST_NAME,
    }
    checkpointer = object()
    resources.background_specialists[SOFTWARE_SPECIALIST_NAME](checkpointer=checkpointer)
    assert created["software_calls"][-1]["unattended"] is True
    assert created["software_calls"][-1]["checkpointer"] is checkpointer
    assert len(created["neutron_calls"]) == 1

    agent_kwargs = created["agent_kwargs"]
    assert agent_kwargs["model"] is supervisor_model
    assert [tool.name for tool in agent_kwargs["tools"]] == [
        "ask_user",
        "start_research",
        "check_research",
        "list_research",
    ]
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
    monkeypatch.setattr(
        specialist_runtime, "create_summarization_middleware", lambda *_args: object()
    )
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
async def test_shutdown_closes_an_instance_that_asks_to_be_closed() -> None:
    """Core's shutdown contract is one method on the instance object.

    It used to reach for `instance.context7_runtime.client` by name, which only
    ever made sense for this application. The replacement is generic: if the
    object an agent factory returned exposes `aclose()` or `close()`, shutdown
    calls it.

    JüNA's own `JuenaAgentResources` exposes neither, and that is now correct
    rather than an omission -- `juena_core.mcp` hands back tools that reopen
    their own connection per call, so Context7 has no client whose lifetime
    must match the agent's. The retired adapter had no `close`/`aclose` either,
    so this shutdown was already doing nothing for it.
    """

    closed: list[str] = []

    class ResourcesThatHoldSomething:
        async def aclose(self) -> None:
            closed.append("closed")

    existing = dict(agent_registry._agent_registry)
    agent_registry._agent_registry.clear()
    agent_registry._agent_registry["test-agent"] = (ResourcesThatHoldSomething(), object())

    try:
        await agent_registry.shutdown_agents()
    finally:
        agent_registry._agent_registry.clear()
        agent_registry._agent_registry.update(existing)

    assert closed == ["closed"]
    assert not hasattr(juena_agent.JuenaAgentResources, "aclose")
