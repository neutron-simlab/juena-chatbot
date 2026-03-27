"""Tests for code-chat agent runtime resource reuse."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from juena.agents import code_chat_agent
from juena.server import agent_registry


@pytest.mark.asyncio
async def test_code_chat_agent_reuses_compiled_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = agent_registry._agent_registry.pop("code_chat_agent", None)
    fake_graph = object()
    created: dict[str, Any] = {"create_deep_agent_calls": 0}
    llm = object()
    local_tools = [object(), object(), object()]
    context7_tools = [object(), object()]
    tavily_tool = object()
    context7_runtime = SimpleNamespace(client=object(), tools=context7_tools)

    class StubRepoManager:
        def __init__(self) -> None:
            self.cache_dir = Path("/tmp/test-repos")
            created["repo_manager"] = self
            created["repo_cache_dir"] = self.cache_dir

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager
            created["vector_index"] = self

    def fake_validate_bootstrap_ready(
        repo_manager: StubRepoManager,
        vector_index: StubVectorIndex,
    ) -> None:
        created["validate_bootstrap_ready"] = (repo_manager, vector_index)

    def fake_create_llm_with_fallback(*, provider: str, model: str) -> object:
        created["llm_args"] = (provider, model)
        return llm

    class StubSparseIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager
            created["sparse_index"] = self

    def fake_build_code_chat_tools(
        repo_manager: StubRepoManager,
        vector_index: StubVectorIndex,
        sparse_index: StubSparseIndex | None = None,
    ) -> list[object]:
        created["tool_builder_args"] = (repo_manager, vector_index, sparse_index)
        return local_tools

    async def fake_load_optional_context7_tools() -> object:
        created["load_optional_context7_tools_calls"] = (
            created.get("load_optional_context7_tools_calls", 0) + 1
        )
        return context7_runtime

    def fake_load_optional_tavily_tool() -> object:
        created["load_optional_tavily_tool_calls"] = (
            created.get("load_optional_tavily_tool_calls", 0) + 1
        )
        return tavily_tool

    class FakeSubagentGraph:
        def invoke(
            self,
            input: Any,
            config: Any | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            created["subagent_invoke"] = {
                "input": input,
                "config": config,
                "kwargs": kwargs,
            }
            return {"messages": []}

        async def ainvoke(
            self,
            input: Any,
            config: Any | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            created["subagent_ainvoke"] = {
                "input": input,
                "config": config,
                "kwargs": kwargs,
            }
            return {"messages": []}

    fake_subagent_graph = FakeSubagentGraph()

    def fake_create_agent(**kwargs: Any) -> object:
        created["subagent_agent_kwargs"] = kwargs
        return fake_subagent_graph

    def fake_create_deep_agent(**kwargs: Any) -> object:
        created["create_deep_agent_calls"] += 1
        created["deep_agent_kwargs"] = kwargs
        return fake_graph

    monkeypatch.setattr(code_chat_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(code_chat_agent, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(code_chat_agent, "RepoSparseIndex", StubSparseIndex)
    monkeypatch.setattr(code_chat_agent, "validate_bootstrap_ready", fake_validate_bootstrap_ready)
    monkeypatch.setattr(code_chat_agent, "create_llm_with_fallback", fake_create_llm_with_fallback)
    monkeypatch.setattr(code_chat_agent, "create_summarization_middleware", lambda *args: object())
    monkeypatch.setattr(code_chat_agent, "build_code_chat_tools", fake_build_code_chat_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_context7_tools", fake_load_optional_context7_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_tavily_tool", fake_load_optional_tavily_tool)
    monkeypatch.setattr(code_chat_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(code_chat_agent, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(code_chat_agent, "get_checkpointer", lambda: object())

    try:
        graph_one = await agent_registry.get_agent("code_chat_agent", provider="openai", model="gpt-test")
        graph_two = await agent_registry.get_agent("code_chat_agent", provider="openai", model="gpt-test")
        agent_resources, _ = agent_registry._agent_registry["code_chat_agent"]
    finally:
        agent_registry._agent_registry.pop("code_chat_agent", None)
        if existing is not None:
            agent_registry._agent_registry["code_chat_agent"] = existing

    assert graph_one is fake_graph
    assert graph_two is fake_graph
    assert isinstance(agent_resources, code_chat_agent.CodeChatAgentResources)
    assert agent_resources.app is fake_graph
    assert agent_resources.context7_runtime is context7_runtime
    validated_repo_manager, validated_vector_index = created["validate_bootstrap_ready"]
    assert validated_repo_manager is validated_vector_index.repo_manager
    tb_rm, tb_vi, tb_si = created["tool_builder_args"]
    assert tb_rm is validated_repo_manager
    assert tb_vi is validated_vector_index
    assert tb_si is created["sparse_index"]
    assert created["load_optional_context7_tools_calls"] == 1
    assert created["load_optional_tavily_tool_calls"] == 1
    assert created["create_deep_agent_calls"] == 1
    assert created["subagent_agent_kwargs"]["model"] is llm
    assert created["subagent_agent_kwargs"]["tools"] == [*local_tools, *context7_tools]
    assert created["deep_agent_kwargs"]["model"] is llm
    assert created["deep_agent_kwargs"]["tools"] == [*context7_tools, tavily_tool]
    backend_factory = created["deep_agent_kwargs"]["backend"]
    assert callable(backend_factory)
    composite_backend = backend_factory(type("Runtime", (), {"state": {}})())
    assert isinstance(composite_backend, code_chat_agent.CompositeBackend)
    assert isinstance(composite_backend.default, code_chat_agent.ReadOnlyStateBackend)
    assert isinstance(composite_backend.routes["/repos/"], code_chat_agent.ReadOnlyFilesystemBackend)
    assert composite_backend.routes["/repos/"].cwd == created["repo_cache_dir"].resolve()
    repo_subagent = created["deep_agent_kwargs"]["subagents"][0]
    assert repo_subagent["name"] == "general-purpose"
    assert repo_subagent["runnable"] is fake_subagent_graph
    repo_subagent["runnable"].invoke({"messages": []})
    assert created["llm_args"] == ("openai", "gpt-test")


@pytest.mark.asyncio
async def test_code_chat_agent_without_context7_uses_only_local_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = agent_registry._agent_registry.pop("code_chat_agent", None)
    fake_graph = object()
    created: dict[str, Any] = {}
    llm = object()
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

    def fake_create_agent(**kwargs: Any) -> object:
        created["subagent_agent_kwargs"] = kwargs
        return object()

    def fake_create_deep_agent(**kwargs: Any) -> object:
        created["deep_agent_kwargs"] = kwargs
        return fake_graph

    class StubSparseIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

    monkeypatch.setattr(code_chat_agent, "RepoManager", StubRepoManager)
    monkeypatch.setattr(code_chat_agent, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(code_chat_agent, "RepoSparseIndex", StubSparseIndex)
    monkeypatch.setattr(code_chat_agent, "validate_bootstrap_ready", lambda *args: None)
    monkeypatch.setattr(code_chat_agent, "create_llm_with_fallback", lambda **kwargs: llm)
    monkeypatch.setattr(code_chat_agent, "create_summarization_middleware", lambda *args: object())
    monkeypatch.setattr(code_chat_agent, "build_code_chat_tools", fake_build_code_chat_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_context7_tools", fake_load_optional_context7_tools)
    monkeypatch.setattr(code_chat_agent, "load_optional_tavily_tool", fake_load_optional_tavily_tool)
    monkeypatch.setattr(code_chat_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(code_chat_agent, "create_deep_agent", fake_create_deep_agent)
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
    assert created["deep_agent_kwargs"]["tools"] == []


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


def test_code_chat_prompts_cover_staged_user_inputs() -> None:
    assert "/inputs/" in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "/inputs/" in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "chat response" in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "instead of editing staged files" in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "For coding questions, inspect local indexed repositories first." in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "Use Context7 tools next" in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "Use Tavily only as a last resort after local repositories and Context7" in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "local repositories first," in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "Context7 second, Tavily only" in code_chat_agent.CODE_CHAT_SYSTEM_PROMPT
    assert "Tavily web search is not available in this subagent" in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "Exhaust local repository evidence before consulting external sources." in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "coordinator can use Tavily web search" in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
    assert "last resort" in code_chat_agent.CODE_CHAT_RESEARCH_SUBAGENT_PROMPT
