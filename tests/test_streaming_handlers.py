"""Tests for streaming handler normalization."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import Overwrite

from juena.server.streaming.handlers import UpdatesStreamHandler


def test_process_updates_unwraps_overwrite_messages() -> None:
    handler = UpdatesStreamHandler(
        agent=object(),  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
        run_id="run-1",
        user_input_message="hello",
    )
    message = AIMessage(content="hello from subagent")

    messages = handler.process_updates(
        {
            "agent": {
                "messages": Overwrite(value=[message]),
            }
        }
    )

    assert messages == [message]


def test_process_updates_normalizes_single_message() -> None:
    handler = UpdatesStreamHandler(
        agent=object(),  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
        run_id="run-1",
        user_input_message="hello",
    )
    message = AIMessage(content="hello once")

    messages = handler.process_updates({"agent": {"messages": message}})

    assert messages == [message]
