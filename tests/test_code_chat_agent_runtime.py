"""Tests for code-chat agent runtime resource reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver

from juena.agents import code_chat_agent
from juena.server import agent_registry
from juena.server.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware


@pytest.mark.asyncio
async def test_code_chat_agent_without_context7_uses_only_local_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = agent_registry._agent_registry.pop("code_chat_agent", None)
    fake_graph = object()
    created: dict[str, Any] = {}
    llm = object()
    llm_subagent = object()
    llm_summarizer = object()
    local_tools = [object(), object()]

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

    def fake_create_llm(*, provider: str, model: str, temperature: float = 0.0, **kwargs: Any) -> object:
        created.setdefault("llm_factory_calls", []).append({
            "provider": provider,
            "model": model,
            "temperature": temperature,
            **kwargs,
        })
        if model == code_chat_agent.CODE_CHAT_SUBAGENT_MODEL:
            return llm_subagent
        if model == code_chat_agent.CODE_CHAT_SUMMARIZER_MODEL:
            return llm_summarizer
        raise AssertionError(f"Unexpected static model request: {model}")

    def fake_create_agent(**kwargs: Any) -> object:
        name = kwargs.get("name")
        if name == "code_chat_repo_research_subagent":
            created["subagent_agent_kwargs"] = kwargs
            return object()
        if name == "code_chat_agent":
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

    monkeypatch.setattr(code_chat_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(code_chat_agent, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(code_chat_agent, "validate_bootstrap_ready", lambda *args: None)
    monkeypatch.setattr(code_chat_agent, "create_llm_with_fallback", lambda **kwargs: llm)
    monkeypatch.setattr(code_chat_agent.LLMFactory, "create_llm", fake_create_llm)
    monkeypatch.setattr(code_chat_agent, "create_summarization_middleware", fake_create_summarization_middleware)
    monkeypatch.setattr(code_chat_agent, "build_code_chat_tools", fake_build_code_chat_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_context7_tools", fake_load_optional_context7_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_tavily_tool", fake_load_optional_tavily_tool)
    monkeypatch.setattr(code_chat_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(code_chat_agent, "get_checkpointer", lambda: object())

    try:
        graph = await agent_registry.get_agent("code_chat_agent", provider="openai", model="gpt-test")
        agent_resources, _ = agent_registry._agent_registry["code_chat_agent"]
    finally:
        agent_registry._agent_registry.pop("code_chat_agent", None)
        if existing is not None:
            agent_registry._agent_registry["code_chat_agent"] = existing

    assert graph is fake_graph
    assert isinstance(agent_resources, code_chat_agent.CodeChatAgentResources)
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
    assert created["summarization_middleware_args"][0][0] is llm_summarizer
    assert created["summarization_middleware_args"][1][0] is llm_summarizer
    assert created["summarization_middleware_args"][1][1] is created["summarization_middleware_args"][0][1]
    assert created["llm_factory_calls"] == [
        {
            "provider": "blablador",
            "model": "alias-code",
            "temperature": 0.0,
        },
        {
            "provider": "blablador",
            "model": "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
            "temperature": 0.0,
        },
    ]
    assert created["coordinator_agent_kwargs"]["tools"] == []
    assert created["coordinator_agent_kwargs"]["state_schema"] is FilesystemState
    assert created["coordinator_agent_kwargs"]["context_schema"] is RuntimeModelContext
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
    assert not any(
        isinstance(middleware, FilesystemMiddleware)
        for middleware in created["coordinator_agent_kwargs"]["middleware"]
    )
    assert created["coordinator_with_config"] == {"recursion_limit": 1000}


def test_read_only_filesystem_backend_rejects_writes(tmp_path: Path) -> None:
    backend = code_chat_agent.ReadOnlyFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    write_result = backend.write("/repo-a/new.txt", "hello")
    edit_result = backend.edit("/repo-a/existing.txt", "old", "new")

    assert write_result.error is not None
    assert "read-only repository path" in write_result.error
    assert edit_result.error is not None
    assert "read-only repository path" in edit_result.error


def test_read_only_state_backend_rejects_writes() -> None:
    runtime = type("Runtime", (), {"state": {"files": {"/inputs/current_code.py": {"content": ["print('x')"]}}}})()
    backend = code_chat_agent.ReadOnlyStateBackend(runtime)

    write_result = backend.write("/inputs/new.txt", "hello")
    edit_result = backend.edit("/inputs/current_code.py", "print('x')", "print('y')")
    scratch_write = backend.write("/scratch.txt", "hello")

    assert write_result.error is not None
    assert "read-only staged input path" in write_result.error
    assert edit_result.error is not None
    assert "read-only staged input path" in edit_result.error
    assert scratch_write.path == "/scratch.txt"


def test_code_chat_backend_exposes_repo_cache_under_repos_prefix(tmp_path: Path) -> None:
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()

    backend_factory = code_chat_agent._build_code_chat_backend(tmp_path)
    composite_backend = backend_factory(type("Runtime", (), {"state": {}})())

    paths = sorted(info["path"] for info in composite_backend.ls_info("/repos"))

    assert paths == ["/repos/repo-a/", "/repos/repo-b/"]


def test_code_chat_backend_exposes_staged_inputs_under_inputs_prefix(tmp_path: Path) -> None:
    runtime = type(
        "Runtime",
        (),
        {
            "state": {
                "files": {
                    "/inputs/current_message.txt": {
                        "content": ["please inspect this file"],
                        "created_at": "c",
                        "modified_at": "m",
                    },
                    "/inputs/uploads/example.py": {
                        "content": ["print('x')"],
                        "created_at": "c",
                        "modified_at": "m",
                    },
                    "/scratch.txt": {
                        "content": ["temporary note"],
                        "created_at": "c",
                        "modified_at": "m",
                    },
                }
            }
        },
    )()
    backend_factory = code_chat_agent._build_code_chat_backend(tmp_path)
    composite_backend = backend_factory(runtime)

    root_paths = sorted(info["path"] for info in composite_backend.ls_info("/"))
    input_paths = sorted(info["path"] for info in composite_backend.ls_info("/inputs"))
    upload_paths = sorted(info["path"] for info in composite_backend.ls_info("/inputs/uploads"))

    assert root_paths == ["/inputs/", "/repos/", "/scratch.txt"]
    assert input_paths == ["/inputs/current_message.txt", "/inputs/uploads/"]
    assert upload_paths == ["/inputs/uploads/example.py"]
    assert "please inspect this file" in composite_backend.read("/inputs/current_message.txt")
    assert "print('x')" in composite_backend.read("/inputs/uploads/example.py")


class _ToolCapableFakeListChatModel(FakeListChatModel):
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "_ToolCapableFakeListChatModel":
        return self


def test_code_chat_coordinator_persists_staged_inputs_in_graph_state() -> None:
    class _UnusedSubagent:
        def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> dict[str, Any]:
            return {"messages": []}

        async def ainvoke(
            self,
            input: Any,
            config: Any | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"messages": []}

    agent = create_agent(
        model=_ToolCapableFakeListChatModel(responses=["done"]),
        tools=[],
        middleware=[
            TodoListMiddleware(),
            SubAgentMiddleware(
                backend=code_chat_agent.ReadOnlyStateBackend,
                subagents=[
                    {
                        "name": code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_NAME,
                        "description": "unused test subagent",
                        "runnable": _UnusedSubagent(),
                    }
                ],
            ),
            create_summarization_middleware(
                _ToolCapableFakeListChatModel(responses=["summary"]),
                code_chat_agent.ReadOnlyStateBackend,
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


