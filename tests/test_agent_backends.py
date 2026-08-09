from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from juena.agents.backends import (
    JUENA_MEMORY_SYSTEM_PROMPT,
    MEMORY_SOURCES,
    ReadOnlyFilesystemBackend,
    ReadOnlyInputsStateBackend,
    SupervisorStateBackend,
    build_supervisor_backend,
    user_store_namespace,
)
from juena.agents.specialist_runtime import build_specialist_backend
from juena.server.agent.runtime_model_middleware import RuntimeModelContext


def test_read_only_filesystem_backend_rejects_writes(tmp_path: Path) -> None:
    backend = ReadOnlyFilesystemBackend(
        root_dir=tmp_path,
        virtual_mode=True,
        label="read-only repository",
    )

    assert backend.write("/new.txt", "hello").error is not None
    assert backend.edit("/existing.txt", "old", "new").error is not None


def test_staged_inputs_are_read_only_for_specialists() -> None:
    backend = ReadOnlyInputsStateBackend()

    assert backend.write("/inputs/uploads/report.csv", "tampered").error is not None
    assert backend.edit("/inputs/uploads/report.csv", "a", "b").error is not None


def test_supervisor_state_backend_hides_staged_and_scratch_files() -> None:
    backend = SupervisorStateBackend()

    assert backend.read("/inputs/current_message.txt").error is not None
    assert backend.write("/scratch.txt", "blocked").error is not None
    assert backend.edit("/inputs/current_message.txt", "a", "b").error is not None
    assert backend.ls("/").entries == []


def test_supervisor_filesystem_filters_the_execute_tool() -> None:
    middleware = FilesystemMiddleware(
        backend=build_supervisor_backend(InMemoryStore()),
    )
    visible_tools: list[str] = []
    request = ModelRequest(model=object(), messages=[], tools=middleware.tools)

    middleware.wrap_model_call(
        request,
        lambda current: visible_tools.extend(tool.name for tool in current.tools)
        or ModelResponse(result=[AIMessage(content="done")]),
    )

    assert "execute" not in visible_tools


def test_specialist_backend_exposes_only_requested_repository_route(tmp_path: Path) -> None:
    (tmp_path / "repo-a").mkdir()
    software_backend = build_specialist_backend(repo_cache_root=tmp_path)
    science_backend = build_specialist_backend()

    repo_listing = software_backend.ls("/repos")
    assert repo_listing.error is None
    assert [entry["path"] for entry in repo_listing.entries or []] == ["/repos/repo-a/"]
    assert "/repos/" not in science_backend.routes


def test_memory_prompt_is_opt_in_and_supports_delegation_preferences() -> None:
    from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT, MemoryMiddleware

    assert "might not explicitly ask you to remember" in MEMORY_SYSTEM_PROMPT
    assert "might not explicitly ask you to remember" not in JUENA_MEMORY_SYSTEM_PROMPT

    prompt = JUENA_MEMORY_SYSTEM_PROMPT.lower()
    assert "only when the user explicitly asks" in prompt
    assert "never infer" in prompt
    assert "never save credentials" in prompt
    assert "untrusted" in prompt
    assert "choose a specialist" in prompt
    assert "never paste the complete memory file" in prompt

    middleware = MemoryMiddleware(
        backend=ReadOnlyInputsStateBackend(),
        sources=MEMORY_SOURCES,
        system_prompt=JUENA_MEMORY_SYSTEM_PROMPT,
    )
    assert middleware.sources == ["/memories/AGENTS.md"]


def test_memory_namespace_uses_trusted_user_uuid() -> None:
    alice = SimpleNamespace(context={"user_id": "00000000-0000-0000-0000-000000000001"})
    bob = SimpleNamespace(context={"user_id": "00000000-0000-0000-0000-000000000002"})

    assert user_store_namespace(alice) != user_store_namespace(bob)
    assert user_store_namespace(alice) == (
        "memories",
        "00000000-0000-0000-0000-000000000001",
    )
    with pytest.raises(ValueError, match="Authenticated user"):
        user_store_namespace(SimpleNamespace(context={}))


class _MemoryState(TypedDict):
    value: str


@pytest.mark.asyncio
async def test_memory_files_are_persisted_per_user() -> None:
    store = InMemoryStore()
    backend = build_supervisor_backend(store)

    async def write_agents_file(state: _MemoryState) -> dict:
        result = await backend.awrite("/memories/AGENTS.md", state["value"])
        assert result.error is None
        return {}

    builder = StateGraph(_MemoryState, context_schema=RuntimeModelContext)
    builder.add_node("write", write_agents_file)
    builder.add_edge(START, "write")
    builder.add_edge("write", END)
    graph = builder.compile(store=store)

    alice = "00000000-0000-0000-0000-000000000001"
    bob = "00000000-0000-0000-0000-000000000002"
    await graph.ainvoke(
        {"value": "# Preferences\n\n- Prefer Python examples.\n"},
        context=RuntimeModelContext(
            provider="openai",
            model="test-model",
            user_id=alice,
        ),
    )

    alice_file = await store.aget(("memories", alice), "/AGENTS.md")
    assert alice_file is not None
    assert "Prefer Python" in alice_file.value["content"]
    assert await store.aget(("memories", bob), "/AGENTS.md") is None


def test_specialist_backend_reads_staged_inputs_inside_graph(tmp_path: Path) -> None:
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    agent = create_agent(
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "read_file",
                        "args": {"file_path": "/inputs/current_message.txt"},
                        "id": "read-input",
                        "type": "tool_call",
                    }],
                ),
                AIMessage(content="done"),
            ]
        ),
        tools=[],
        middleware=[FilesystemMiddleware(backend=build_specialist_backend())],
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {
            "messages": [("user", "inspect the staged input")],
            "files": {"/inputs/current_message.txt": create_file_data("supplied evidence")},
        },
        config={"configurable": {"thread_id": "staged-specialist"}},
    )

    tool_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-input"
    )
    assert "supplied evidence" in str(tool_message.content)
