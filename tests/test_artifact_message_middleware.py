"""Attaching generated files to the answer, including when a run is cut short."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from juena_core.agents.loop_guard import RepeatedToolCallMiddleware
from juena_core.artifacts import (
    ARTIFACT_MESSAGE_KEY,
    ArtifactStore,
    set_artifact_store_for_tests,
)
from juena_core.sandbox.middleware import ArtifactMessageMiddleware


def _png() -> bytes:
    """Smallest PNG the store will accept as an image."""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def store(tmp_path) -> Any:  # noqa: ANN001
    value = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(value)
    yield value
    set_artifact_store_for_tests(None)


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(context={"user_id": "user-a", "thread_id": "thread-a"})


def test_artifacts_attach_to_the_final_answer(store: Any) -> None:
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="fit.png",
        content=_png(),
    )
    answer = AIMessage(content="Here is the fit.")

    update = ArtifactMessageMiddleware().after_agent(
        {"messages": [answer]}, _runtime()
    )

    assert update is not None
    assert [item["filename"] for item in update["messages"][0].additional_kwargs[ARTIFACT_MESSAGE_KEY]] == [
        "fit.png"
    ]


def test_nothing_is_claimed_while_the_model_is_still_calling_tools(store: Any) -> None:
    """A message with pending tool calls is not the answer yet.

    Claiming is destructive -- `claim_for_message` empties the queue -- so an
    early claim would attach the plot to a message the user never sees.
    """
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="fit.png",
        content=_png(),
    )
    mid_turn = AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {}, "id": "x", "type": "tool_call"}],
    )

    assert ArtifactMessageMiddleware().after_agent({"messages": [mid_turn]}, _runtime()) is None
    assert store.claim_for_message("user-a", "thread-a") != []


@pytest.mark.asyncio
async def test_artifacts_survive_a_run_that_a_budget_cut_short(store: Any) -> None:
    """The regression that moved this hook from `after_model` to `after_agent`.

    A budget that ends a run returns `jump_to: "end"`, and the graph routes
    straight past every `after_model` hook. The supervisor has no other
    `after_agent`, so its exit node was `END` -- and a turn cut short dropped its
    artifacts silently: the plot was generated, collected, and then attached to
    nothing. Part 1 makes that path common, so it has to hold.
    """
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool as make_tool

    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="fit.png",
        content=_png(),
    )

    @make_tool
    def read_file(file_path: str) -> str:
        """Read a file."""
        return "line 1"

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
                        "args": {"file_path": "/a.dat"},
                        "id": f"call-{index}",
                        "type": "tool_call",
                    }],
                )
                for index in range(12)
            ]
        ),
        tools=[read_file],
        middleware=[RepeatedToolCallMiddleware(), ArtifactMessageMiddleware()],
        context_schema=dict,
    )
    result = await agent.ainvoke(
        {"messages": [("user", "read it")]},
        context={"user_id": "user-a", "thread_id": "thread-a"},
    )

    final = result["messages"][-1]
    assert "Stopped after" in str(final.content), "the run did not take the cut-short path"
    assert [item["filename"] for item in final.additional_kwargs[ARTIFACT_MESSAGE_KEY]] == [
        "fit.png"
    ]
