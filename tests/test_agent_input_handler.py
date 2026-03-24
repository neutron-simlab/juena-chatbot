"""Tests for agent input preparation."""

from __future__ import annotations

from uuid import UUID

import pytest

from juena.server.agent_input_handler import AgentInputHandler


@pytest.mark.asyncio
async def test_prepare_input_builds_standard_langgraph_kwargs() -> None:
    kwargs, run_id = await AgentInputHandler.prepare_input(
        "hello",
        thread_id="thread-1",
        user_id="user-1",
        provider="openai",
        model="gpt-test",
    )

    assert isinstance(run_id, UUID)
    assert "context" not in kwargs
    assert kwargs["input"]["messages"][0].content == "hello"
    assert kwargs["input"]["thread_id"] == "thread-1"
    assert kwargs["input"]["user_id"] == "user-1"
    assert kwargs["config"]["configurable"]["provider"] == "openai"
    assert kwargs["config"]["configurable"]["model"] == "gpt-test"
