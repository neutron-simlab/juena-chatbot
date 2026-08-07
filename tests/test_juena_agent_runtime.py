"""Tests for code-chat agent runtime resource reuse."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends.utils import create_file_data, file_data_to_string
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, TodoListMiddleware
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from juena.agents import juena_agent
from juena.server import agent_registry
from juena.server.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware


def test_code_chat_prompts_require_subagent_research_packets() -> None:
    prompt = juena_agent.build_system_prompt(
        has_repo_research=True, has_context7=True, has_tavily=True
    )
    assert "structured research packets" in prompt
    assert "write the user-facing" in prompt
    assert 'repo_id="all"' in prompt
    assert "filesystem tools only to verify" in prompt
    assert "Indexed search previews are leads" in prompt
    assert "structured research packet" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Use indexed search as the primary" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Do not use filesystem tools as broad search" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Evidence ledger" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Recommended final answer points" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "`grep` only" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "`glob` only" in juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Grep is exact verification" in juena_agent.CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS["grep"]
    assert "not semantic discovery" in juena_agent.CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS["grep"]
    assert "indexed search is the primary" in juena_agent.CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS["ls"]


def test_system_prompt_omits_sections_for_absent_capabilities() -> None:
    """Instructing the model to use a tool it was not given causes refusals and
    invented tool names, so absent capabilities must drop out of the prompt."""
    bare = juena_agent.build_system_prompt(
        has_repo_research=False, has_context7=False, has_tavily=False
    )
    assert "repository index is unavailable" in bare
    assert "Tavily" not in bare
    assert "Context7" not in bare
    assert juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_NAME not in bare

    web_only = juena_agent.build_system_prompt(
        has_repo_research=False, has_context7=False, has_tavily=True
    )
    assert "Tavily" in web_only
    assert "Context7" not in web_only

    # Rules stay contiguously numbered no matter which sections are dropped.
    numbers = [
        int(line.split(".", 1)[0])
        for line in bare.splitlines()
        if line[:1].isdigit()
    ]
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.asyncio
async def test_agent_builds_without_repo_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing repository index must not take down the whole agent: JueNA also
    answers science questions that never touch a repository."""
    existing = agent_registry._agent_registry.pop(juena_agent.JUENA_AGENT_ID, None)
    created: dict[str, Any] = {}
    fake_graph = object()

    class StubRepoManager:
        def __init__(self) -> None:
            self.cache_dir = Path("/tmp/test-repos")

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

    def fake_create_agent(**kwargs: Any) -> object:
        created["kwargs"] = kwargs

        class FakeGraph:
            def with_config(self, config: dict[str, Any]) -> object:
                return fake_graph

        return FakeGraph()

    async def fake_load_optional_context7_tools() -> None:
        return None

    monkeypatch.setattr(juena_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(juena_agent, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(
        juena_agent,
        "describe_bootstrap_problems",
        lambda *args: ["vector index missing for 'silx'"],
    )
    monkeypatch.setattr(juena_agent, "build_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(
        juena_agent,
        "create_summarization_middleware",
        lambda model, backend: object(),
    )
    monkeypatch.setattr(juena_agent, "load_optional_context7_tools", fake_load_optional_context7_tools)
    monkeypatch.setattr(juena_agent, "load_optional_tavily_tool", lambda: None)
    monkeypatch.setattr(juena_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(juena_agent, "get_checkpointer", lambda: object())
    monkeypatch.setattr(juena_agent, "get_store", lambda: object())

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("repo tools must not be built without an index")

    monkeypatch.setattr(juena_agent, "build_code_chat_tools", unexpected)

    try:
        graph = await agent_registry.get_agent(
            juena_agent.JUENA_AGENT_ID, provider="openai", model="gpt-test"
        )
    finally:
        agent_registry._agent_registry.pop(juena_agent.JUENA_AGENT_ID, None)
        if existing is not None:
            agent_registry._agent_registry[juena_agent.JUENA_AGENT_ID] = existing

    assert graph is fake_graph
    # No repo research means no subagent middleware and no repo instructions.
    assert not any(
        isinstance(middleware, SubAgentMiddleware)
        for middleware in created["kwargs"]["middleware"]
    )
    assert "repository index is unavailable" in created["kwargs"]["system_prompt"]


def test_fallback_models_do_not_depend_on_initial_primary_provider(monkeypatch) -> None:
    """The first request must not decide fallback behavior for the cached graph."""
    built: list[dict[str, str]] = []
    sentinel = object()

    monkeypatch.setattr(juena_agent.global_config, "FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(juena_agent, "get_available_providers", lambda: {"openai": True})
    monkeypatch.setattr(juena_agent, "get_default_model", lambda _p: "gpt-4o-mini")
    monkeypatch.setattr(
        juena_agent,
        "build_chat_model",
        lambda **kwargs: built.append(kwargs) or sentinel,
    )

    assert juena_agent.build_fallback_models() == [sentinel]
    assert built == [{"provider": "openai", "model": "gpt-4o-mini"}]


def test_fallback_models_skipped_when_provider_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(juena_agent.global_config, "FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(juena_agent, "get_available_providers", lambda: {"openai": False})
    assert juena_agent.build_fallback_models() == []


def test_fallback_models_built_for_distinct_available_provider(monkeypatch) -> None:
    built: list[dict[str, str]] = []
    sentinel = object()

    monkeypatch.setattr(juena_agent.global_config, "FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(juena_agent, "get_available_providers", lambda: {"openai": True})
    monkeypatch.setattr(juena_agent, "get_default_model", lambda _p: "gpt-4o-mini")
    monkeypatch.setattr(
        juena_agent,
        "build_chat_model",
        lambda **kwargs: built.append(kwargs) or sentinel,
    )

    assert juena_agent.build_fallback_models() == [sentinel]
    assert built == [{"provider": "openai", "model": "gpt-4o-mini"}]


def test_runtime_model_middleware_wraps_outside_model_fallback(monkeypatch) -> None:
    """Wrap hooks nest first-outermost. If the fallback ran outside the runtime
    override, the override would reinstate the just-failed model on every
    fallback attempt and the agent would retry the same broken model forever."""
    middleware = juena_agent._resilience_middleware(
        [object()], model_call_limit=10, tool_call_limit=20
    )
    types = [type(m).__name__ for m in middleware]

    assert types.index("ModelFallbackMiddleware") < types.index("ModelRetryMiddleware"), (
        "fallback must wrap retry so a model is retried before being swapped"
    )
    assert "ModelCallLimitMiddleware" in types
    assert "ToolCallLimitMiddleware" in types
    assert "ToolRetryMiddleware" in types

    retry = next(m for m in middleware if type(m).__name__ == "ModelRetryMiddleware")
    assert retry.max_retries == 3
    assert retry.on_failure == "error"


def test_model_fallback_runs_after_retry_exhaustion(monkeypatch) -> None:
    primary = object()
    fallback = object()
    request = ModelRequest(model=primary, messages=[])
    middleware = juena_agent._resilience_middleware(
        [fallback], model_call_limit=10, tool_call_limit=20
    )
    fallback_middleware = next(
        m for m in middleware if type(m).__name__ == "ModelFallbackMiddleware"
    )
    retry_middleware = next(
        m for m in middleware if type(m).__name__ == "ModelRetryMiddleware"
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
        lambda current_request: retry_middleware.wrap_model_call(
            current_request, call_model
        ),
    )

    assert attempts == [primary, primary, primary, primary, fallback]
    assert response.result[0].text == "fallback succeeded"


def test_resilience_middleware_omits_fallback_without_models() -> None:
    types = [
        type(m).__name__
        for m in juena_agent._resilience_middleware([], model_call_limit=1, tool_call_limit=1)
    ]
    assert "ModelFallbackMiddleware" not in types


def test_legacy_agent_ids_still_resolve() -> None:
    assert agent_registry._resolve_agent_id("code_chat_agent") == juena_agent.JUENA_AGENT_ID
    assert agent_registry._resolve_agent_id("react_agent") == juena_agent.JUENA_AGENT_ID
    assert agent_registry._resolve_agent_id("juena") == juena_agent.JUENA_AGENT_ID
    # Aliases are resolvable but not advertised as separate agents.
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
        SimpleNamespace(
            context7_runtime=SimpleNamespace(client=FakeClient()),
        ),
        object(),
    )

    try:
        await agent_registry.shutdown_agents()
    finally:
        agent_registry._agent_registry.clear()
        agent_registry._agent_registry.update(existing)

    assert closed == [True]


@pytest.mark.asyncio
async def test_juena_agent_without_context7_uses_only_local_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = agent_registry._agent_registry.pop(juena_agent.JUENA_AGENT_ID, None)
    fake_graph = object()
    created: dict[str, Any] = {}
    llm = object()
    llm_subagent = object()
    llm_summarizer = object()
    local_tools = [object(), object()]
    store = object()

    class StubRepoManager:
        def __init__(self) -> None:
            self.cache_dir = Path("/tmp/test-repos")

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

    async def fake_load_optional_context7_tools() -> None:
        return None

    def fake_load_optional_tavily_tool() -> None:
        return None

    def fake_build_code_chat_tools(*args: Any, **kwargs: Any) -> list[object]:
        return local_tools

    def fake_build_chat_model(*, provider: str, model: str, temperature: float = 0.0, **kwargs: Any) -> object:
        created.setdefault("llm_factory_calls", []).append({
            "provider": provider,
            "model": model,
            "temperature": temperature,
            **kwargs,
        })
        if model == juena_agent.CODE_CHAT_SUBAGENT_MODEL:
            return llm_subagent
        if model == juena_agent.CODE_CHAT_SUMMARIZER_MODEL:
            return llm_summarizer
        # Anything else is the user-selected coordinator model.
        return llm

    def fake_create_agent(**kwargs: Any) -> object:
        name = kwargs.get("name")
        if name == "code_chat_repo_research_subagent":
            created["subagent_agent_kwargs"] = kwargs
            return RunnableLambda(lambda _input: {"messages": []})
        if name == juena_agent.JUENA_AGENT_ID:
            created["coordinator_agent_kwargs"] = kwargs

            class FakeCoordinatorGraph:
                def with_config(self, config: dict[str, Any]) -> object:
                    created["coordinator_with_config"] = config
                    return fake_graph

            return FakeCoordinatorGraph()
        raise AssertionError(f"Unexpected agent name: {name}")

    def fake_create_summarization_middleware(model: object, backend: object) -> object:
        created.setdefault("summarization_middleware_args", []).append((model, backend))
        return object()

    monkeypatch.setattr(juena_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(juena_agent, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(juena_agent, "describe_bootstrap_problems", lambda *args: [])
    monkeypatch.setattr(juena_agent, "build_chat_model", fake_build_chat_model)
    # Fallback wiring has its own tests; keep this one independent of .env.
    monkeypatch.setattr(juena_agent, "build_fallback_models", lambda: [])
    monkeypatch.setattr(juena_agent, "create_summarization_middleware", fake_create_summarization_middleware)
    monkeypatch.setattr(juena_agent, "build_code_chat_tools", fake_build_code_chat_tools)
    monkeypatch.setattr(juena_agent, "load_optional_context7_tools", fake_load_optional_context7_tools)
    monkeypatch.setattr(juena_agent, "load_optional_tavily_tool", fake_load_optional_tavily_tool)
    monkeypatch.setattr(juena_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(juena_agent, "get_checkpointer", lambda: object())
    monkeypatch.setattr(juena_agent, "get_store", lambda: store)

    try:
        graph = await agent_registry.get_agent(juena_agent.JUENA_AGENT_ID, provider="openai", model="gpt-test")
        agent_resources, _ = agent_registry._agent_registry[juena_agent.JUENA_AGENT_ID]
    finally:
        agent_registry._agent_registry.pop(juena_agent.JUENA_AGENT_ID, None)
        if existing is not None:
            agent_registry._agent_registry[juena_agent.JUENA_AGENT_ID] = existing

    assert graph is fake_graph
    assert isinstance(agent_resources, juena_agent.CodeChatAgentResources)
    assert agent_resources.context7_runtime is None
    assert created["subagent_agent_kwargs"]["tools"] == local_tools
    assert created["subagent_agent_kwargs"]["context_schema"] is RuntimeModelContext
    assert created["subagent_agent_kwargs"]["model"] is llm_subagent
    assert not any(
        isinstance(middleware, RuntimeModelMiddleware)
        for middleware in created["subagent_agent_kwargs"]["middleware"]
    )
    assert any(
        isinstance(middleware, FilesystemMiddleware)
        for middleware in created["subagent_agent_kwargs"]["middleware"]
    )
    filesystem_middleware = next(
        middleware
        for middleware in created["subagent_agent_kwargs"]["middleware"]
        if isinstance(middleware, FilesystemMiddleware)
    )
    assert filesystem_middleware._custom_tool_descriptions == juena_agent.CODE_CHAT_FILESYSTEM_TOOL_DESCRIPTIONS
    assert created["summarization_middleware_args"][0][0] is llm_summarizer
    assert created["summarization_middleware_args"][1][0] is llm_summarizer
    assert created["summarization_middleware_args"][1][1] is created["summarization_middleware_args"][0][1]
    # The coordinator uses the user-selected provider/model; the subagent and
    # summarizer are pinned to their static Blablador models.
    assert created["llm_factory_calls"] == [
        {
            "provider": "openai",
            "model": "gpt-test",
            "temperature": 0.0,
        },
        {
            "provider": "blablador",
            "model": juena_agent.CODE_CHAT_SUMMARIZER_MODEL,
            "temperature": 0.0,
        },
        {
            "provider": "blablador",
            "model": juena_agent.CODE_CHAT_SUBAGENT_MODEL,
            "temperature": 0.0,
        },
    ]
    assert created["coordinator_agent_kwargs"]["tools"] == []
    assert created["coordinator_agent_kwargs"]["store"] is store
    # The coordinator does not pass state_schema explicitly: FilesystemMiddleware
    # declares FilesystemState and create_agent merges middleware state schemas.
    assert "state_schema" not in created["coordinator_agent_kwargs"]
    assert FilesystemMiddleware.state_schema is FilesystemState
    assert created["coordinator_agent_kwargs"]["context_schema"] is RuntimeModelContext
    coordinator_filesystem = next(
        middleware
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
        if isinstance(middleware, FilesystemMiddleware)
    )
    # The coordinator's filesystem tools serve skills and memory only; repo and
    # /inputs reading stays delegated to the subagent.
    assert (
        coordinator_filesystem._custom_tool_descriptions
        == juena_agent.CODE_CHAT_COORDINATOR_FILESYSTEM_TOOL_DESCRIPTIONS
    )
    assert "code-chat-expert" in coordinator_filesystem._custom_tool_descriptions["read_file"]
    assert any(
        isinstance(middleware, TodoListMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert any(
        isinstance(middleware, SubAgentMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    subagent_middleware = next(
        middleware
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
        if isinstance(middleware, SubAgentMiddleware)
    )
    assert subagent_middleware._backend is created["summarization_middleware_args"][0][1]
    assert any(
        isinstance(middleware, RuntimeModelMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert any(
        isinstance(middleware, FilesystemMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert any(
        isinstance(middleware, SkillsMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert any(
        isinstance(middleware, MemoryMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert created["coordinator_with_config"] == {"recursion_limit": 1000}


def test_read_only_filesystem_backend_rejects_writes(tmp_path: Path) -> None:
    backend = juena_agent.ReadOnlyFilesystemBackend(
        root_dir=tmp_path, virtual_mode=True, label="read-only repository"
    )

    write_result = backend.write("/repo-a/new.txt", "hello")
    edit_result = backend.edit("/repo-a/existing.txt", "old", "new")

    assert write_result.error is not None
    assert "read-only repository path" in write_result.error
    assert edit_result.error is not None
    assert "read-only repository path" in edit_result.error


def test_read_only_state_backend_rejects_input_writes_without_runtime() -> None:
    backend = juena_agent.ReadOnlyInputsStateBackend()

    write_result = backend.write("/inputs/new.txt", "hello")
    edit_result = backend.edit("/inputs/current_code.py", "print('x')", "print('y')")

    assert write_result.error is not None
    assert "read-only staged input path" in write_result.error
    assert edit_result.error is not None
    assert "read-only staged input path" in edit_result.error


def test_code_chat_backend_exposes_repo_cache_under_repos_prefix(tmp_path: Path) -> None:
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()

    composite_backend = juena_agent._build_code_chat_backend(tmp_path)
    result = composite_backend.ls("/repos")

    assert result.error is None
    paths = sorted(info["path"] for info in result.entries or [])

    assert paths == ["/repos/repo-a/", "/repos/repo-b/"]


class _ToolCapableFakeMessagesListChatModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> "_ToolCapableFakeMessagesListChatModel":
        return self


def _create_filesystem_test_agent(
    backend: Any,
    responses: list[AIMessage],
) -> Any:
    return create_agent(
        model=_ToolCapableFakeMessagesListChatModel(responses=responses),
        tools=[],
        middleware=[FilesystemMiddleware(backend=backend)],
        state_schema=FilesystemState,
        checkpointer=InMemorySaver(),
    )


def test_code_chat_backend_exposes_and_reads_staged_inputs(tmp_path: Path) -> None:
    backend = juena_agent._build_code_chat_backend(tmp_path)
    agent = _create_filesystem_test_agent(
        backend,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "ls", "args": {"path": "/"}, "id": "ls-root"},
                    {"name": "ls", "args": {"path": "/inputs"}, "id": "ls-inputs"},
                    {
                        "name": "read_file",
                        "args": {"file_path": "/inputs/current_message.txt"},
                        "id": "read-message",
                    },
                    {
                        "name": "read_file",
                        "args": {"file_path": "/inputs/uploads/example.py"},
                        "id": "read-upload",
                    },
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    config = {"configurable": {"thread_id": "staged-inputs", "user_id": "user-1"}}
    staged_files = {
        "/inputs/current_message.txt": create_file_data("please inspect this file"),
        "/inputs/uploads/example.py": create_file_data("print('x')"),
        "/scratch.txt": create_file_data("temporary note"),
    }

    result = agent.invoke(
        {"messages": [("user", "inspect the staged files")], "files": staged_files},
        config=config,
    )

    tool_messages = {
        message.tool_call_id: str(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert "/inputs/" in tool_messages["ls-root"]
    assert "/repos/" in tool_messages["ls-root"]
    assert "/scratch.txt" in tool_messages["ls-root"]
    assert "/inputs/current_message.txt" in tool_messages["ls-inputs"]
    assert "/inputs/uploads/" in tool_messages["ls-inputs"]
    assert "please inspect this file" in tool_messages["read-message"]
    assert "print('x')" in tool_messages["read-upload"]


def test_code_chat_backend_allows_scratch_writes_and_rejects_input_writes(
    tmp_path: Path,
) -> None:
    backend = juena_agent._build_code_chat_backend(tmp_path)
    agent = _create_filesystem_test_agent(
        backend,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/scratch.txt", "content": "temporary note"},
                        "id": "write-scratch",
                    },
                    {
                        "name": "write_file",
                        "args": {"file_path": "/inputs/new.txt", "content": "blocked"},
                        "id": "write-input",
                    },
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    config = {"configurable": {"thread_id": "write-policy", "user_id": "user-1"}}

    result = agent.invoke(
        {"messages": [("user", "write the files")]},
        config=config,
    )
    state = agent.get_state(config)
    values = getattr(state, "values", {}) or {}
    tool_messages = {
        message.tool_call_id: message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }

    assert tool_messages["write-scratch"].status == "success"
    assert tool_messages["write-input"].status == "error"
    assert "read-only staged input path" in str(tool_messages["write-input"].content)
    assert file_data_to_string(values["files"]["/scratch.txt"]) == "temporary note"
    assert "/inputs/new.txt" not in values["files"]


class _ToolCapableFakeListChatModel(FakeListChatModel):
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "_ToolCapableFakeListChatModel":
        return self


def test_code_chat_coordinator_persists_staged_inputs_in_graph_state() -> None:
    backend = juena_agent.ReadOnlyInputsStateBackend()

    agent = create_agent(
        model=_ToolCapableFakeListChatModel(responses=["done"]),
        tools=[],
        middleware=[
            TodoListMiddleware(),
            SubAgentMiddleware(
                backend=backend,
                subagents=[
                    {
                        "name": juena_agent.CODE_CHAT_RESEARCH_SUBAGENT_NAME,
                        "description": "unused test subagent",
                        "runnable": RunnableLambda(lambda _input: {"messages": []}),
                    }
                ],
            ),
            create_summarization_middleware(
                _ToolCapableFakeListChatModel(responses=["summary"]),
                backend,
            ),
        ],
        state_schema=FilesystemState,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread-1", "user_id": "user-1"}}
    staged_files = {
        "/inputs/uploads/p_r_simulation.py": {
            "content": ["print('uploaded')"],
            "created_at": "c",
            "modified_at": "m",
        }
    }

    agent.invoke(
        {
            "messages": [("user", "inspect the uploaded file")],
            "files": staged_files,
        },
        config=config,
    )

    state = agent.get_state(config)
    values = getattr(state, "values", {}) or {}

    assert values["files"] == staged_files


def test_code_chat_backend_reads_legacy_checkpoint_file_data(tmp_path: Path) -> None:
    backend = juena_agent._build_code_chat_backend(tmp_path)
    agent = _create_filesystem_test_agent(
        backend,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/inputs/uploads/legacy.py"},
                        "id": "read-legacy",
                    }
                ],
            ),
            AIMessage(content="done"),
        ],
    )
    config = {"configurable": {"thread_id": "legacy-files", "user_id": "user-1"}}
    legacy_files = {
        "/inputs/uploads/legacy.py": {
            "content": ["print('legacy')", "print('checkpoint')"],
            "created_at": "c",
            "modified_at": "m",
        }
    }

    with pytest.warns(LangChainDeprecationWarning, match=r"list\[str\].*content"):
        result = agent.invoke(
            {"messages": [("user", "read the legacy upload")], "files": legacy_files},
            config=config,
        )

    tool_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-legacy"
    )
    assert "print('legacy')" in str(tool_message.content)
    assert "print('checkpoint')" in str(tool_message.content)
