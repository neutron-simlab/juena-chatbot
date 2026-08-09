"""Tests for agent input preparation."""

from __future__ import annotations

from uuid import UUID

from juena.server.agent.input_handler import AgentInputHandler
from juena.server.agent.runtime_model_middleware import RuntimeModelContext


def test_build_run_context_puts_identity_in_config_and_context() -> None:
    run = AgentInputHandler.build_run_context(
        thread_id="thread-1",
        user_id="user-1",
        provider="openai",
        model="gpt-test",
    )

    assert isinstance(run.run_id, UUID)
    assert run.config["configurable"]["thread_id"] == "thread-1"
    assert run.config["configurable"]["user_id"] == "user-1"
    assert run.config["configurable"]["provider"] == "openai"
    assert run.config["configurable"]["model"] == "gpt-test"
    assert run.context == RuntimeModelContext(
        provider="openai",
        model="gpt-test",
        thread_id="thread-1",
        user_id="user-1",
    )


def test_build_input_carries_only_message_channels() -> None:
    input_data = AgentInputHandler.build_input("hello")

    assert input_data["messages"][0].content == "hello"
    # thread_id/user_id are not state channels: LangGraph drops unknown input
    # keys and logs a warning for each on every invocation. They travel in
    # config.configurable and the runtime context instead.
    assert set(input_data) == {"messages"}


def test_build_input_supports_message_override_and_initial_files() -> None:
    input_data = AgentInputHandler.build_input(
        "raw prompt",
        message_override="inspect /inputs first",
        initial_files={
            "/inputs/current_message.txt": {"content": ["raw prompt"], "created_at": "c", "modified_at": "m"},
            "/inputs/old.txt": None,
        },
    )

    assert input_data["messages"][0].content == "inspect /inputs first"
    assert input_data["files"] == {
        "/inputs/current_message.txt": {"content": ["raw prompt"], "created_at": "c", "modified_at": "m"},
        "/inputs/old.txt": None,
    }
