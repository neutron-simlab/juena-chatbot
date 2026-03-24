"""Tests for code-chat agent runtime resource reuse."""

from __future__ import annotations

from pathlib import Path
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
    tools = [object(), object(), object()]

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

    def fake_build_code_chat_tools(repo_manager: StubRepoManager, vector_index: StubVectorIndex) -> list[object]:
        created["tool_builder_args"] = (repo_manager, vector_index)
        return tools

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
    monkeypatch.setattr(code_chat_agent, "validate_bootstrap_ready", fake_validate_bootstrap_ready)
    monkeypatch.setattr(code_chat_agent, "create_llm_with_fallback", fake_create_llm_with_fallback)
    monkeypatch.setattr(code_chat_agent, "create_summarization_middleware", lambda *args: object())
    monkeypatch.setattr(code_chat_agent, "build_code_chat_tools", fake_build_code_chat_tools)
    monkeypatch.setattr(code_chat_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(code_chat_agent, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(code_chat_agent, "get_checkpointer", lambda: object())

    try:
        graph_one = await agent_registry.get_agent("code_chat_agent", provider="openai", model="gpt-test")
        graph_two = await agent_registry.get_agent("code_chat_agent", provider="openai", model="gpt-test")
    finally:
        agent_registry._agent_registry.pop("code_chat_agent", None)
        if existing is not None:
            agent_registry._agent_registry["code_chat_agent"] = existing

    assert graph_one is fake_graph
    assert graph_two is fake_graph
    validated_repo_manager, validated_vector_index = created["validate_bootstrap_ready"]
    assert validated_repo_manager is validated_vector_index.repo_manager
    assert created["tool_builder_args"] == (validated_repo_manager, validated_vector_index)
    assert created["create_deep_agent_calls"] == 1
    assert created["subagent_agent_kwargs"]["model"] is llm
    assert created["subagent_agent_kwargs"]["tools"] is tools
    assert created["deep_agent_kwargs"]["model"] is llm
    assert created["deep_agent_kwargs"]["tools"] == []
    backend_factory = created["deep_agent_kwargs"]["backend"]
    assert callable(backend_factory)
    composite_backend = backend_factory(type("Runtime", (), {"state": {}})())
    assert isinstance(composite_backend, code_chat_agent.CompositeBackend)
    assert isinstance(composite_backend.default, code_chat_agent.StateBackend)
    assert isinstance(composite_backend.routes["/repos/"], code_chat_agent.ReadOnlyFilesystemBackend)
    assert composite_backend.routes["/repos/"].cwd == created["repo_cache_dir"].resolve()
    repo_subagent = created["deep_agent_kwargs"]["subagents"][0]
    assert repo_subagent["name"] == "general-purpose"
    assert repo_subagent["runnable"] is fake_subagent_graph
    repo_subagent["runnable"].invoke({"messages": []})
    assert created["llm_args"] == ("openai", "gpt-test")


def test_read_only_filesystem_backend_rejects_writes(tmp_path: Path) -> None:
    backend = code_chat_agent.ReadOnlyFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    write_result = backend.write("/repo-a/new.txt", "hello")
    edit_result = backend.edit("/repo-a/existing.txt", "old", "new")

    assert write_result.error is not None
    assert "read-only repository path" in write_result.error
    assert edit_result.error is not None
    assert "read-only repository path" in edit_result.error


def test_code_chat_backend_exposes_repo_cache_under_repos_prefix(tmp_path: Path) -> None:
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()

    backend_factory = code_chat_agent._build_code_chat_backend(tmp_path)
    composite_backend = backend_factory(type("Runtime", (), {"state": {}})())

    paths = sorted(info["path"] for info in composite_backend.ls_info("/repos"))

    assert paths == ["/repos/repo-a/", "/repos/repo-b/"]
