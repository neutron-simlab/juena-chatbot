"""Tests for stream event handling and the supervisor/subagent split."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Interrupt, Overwrite

from juena.server.streaming import events
from juena.server.streaming.handlers import (
    extract_update_messages,
    is_subagent_namespace,
    token_text,
)
from juena.server.streaming.processor import StreamEventProcessor


def _processor(**kwargs: Any) -> StreamEventProcessor:
    return StreamEventProcessor(
        agent=object(),
        config={},
        run_id="run-1",
        user_input_message="hello",
        **kwargs,
    )


async def _collect(processor: StreamEventProcessor, event: Any) -> list[dict[str, Any]]:
    return [payload async for payload in processor.process_event(event)]


# --------------------------------------------------------------------------
# update payload normalization
# --------------------------------------------------------------------------

def test_extract_update_messages_unwraps_overwrite() -> None:
    message = AIMessage(content="hello from subagent")
    assert extract_update_messages({"agent": {"messages": Overwrite(value=[message])}}) == [message]


def test_extract_update_messages_normalizes_single_message() -> None:
    message = AIMessage(content="hello once")
    assert extract_update_messages({"agent": {"messages": message}}) == [message]


def test_extract_update_messages_tolerates_empty_payloads() -> None:
    assert extract_update_messages({}) == []
    assert extract_update_messages({"agent": None}) == []
    assert extract_update_messages({"agent": {"messages": None}}) == []


def test_approval_description_omits_duplicated_tool_arguments() -> None:
    payload = events.approval_required_event(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "python plot.py"},
                    "description": "Review this command.\n\nTool: execute\nArgs: {...}",
                }
            ],
            "review_configs": [{"allowed_decisions": ["approve"]}],
        },
    )

    assert payload["description"] == "Review this command."


# --------------------------------------------------------------------------
# namespace classification
# --------------------------------------------------------------------------

def test_subagent_namespace_detection() -> None:
    assert is_subagent_namespace(("tools:abc123",)) is True
    assert is_subagent_namespace(("agent", "tools:abc123", "model")) is True
    assert is_subagent_namespace(("model",)) is False
    assert is_subagent_namespace(()) is False
    assert is_subagent_namespace(None) is False


# --------------------------------------------------------------------------
# token extraction
# --------------------------------------------------------------------------

def test_token_text_skips_non_chunks_and_tagged_streams() -> None:
    assert token_text((AIMessage(content="complete"), {})) is None
    assert token_text((AIMessageChunk(content="x"), {"tags": ["skip_stream"]})) is None
    assert token_text((AIMessageChunk(content=""), {})) is None


def test_token_text_returns_only_text_blocks() -> None:
    chunk = AIMessageChunk(
        content=[
            {"type": "reasoning", "reasoning": "internal"},
            {"type": "text", "text": "visible"},
        ]
    )
    assert token_text((chunk, {})) == "visible"


# --------------------------------------------------------------------------
# the two regressions this design exists to prevent
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_tokens_never_reach_the_answer_stream() -> None:
    """Subagent tokens are internal reasoning. Streaming them made the UI show
    text that was then replaced by the supervisor's real answer."""
    payloads = await _collect(
        _processor(),
        (("tools:abc",), "messages", (AIMessageChunk(content="thinking out loud"), {})),
    )

    assert all(payload["type"] != "token" for payload in payloads)
    assert payloads[0]["type"] == "status"
    assert payloads[0]["phase"] == events.PHASE_RESEARCHING


@pytest.mark.asyncio
async def test_supervisor_tokens_are_streamed() -> None:
    payloads = await _collect(
        _processor(),
        (("model",), "messages", (AIMessageChunk(content="the answer"), {})),
    )

    assert {"type": "token", "content": "the answer"} in payloads


@pytest.mark.asyncio
async def test_v2_interrupt_becomes_clickable_approval_event() -> None:
    interrupt = Interrupt(
        id="interrupt-1",
        value={
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "python plot.py", "timeout": 600},
                    "description": "Create a plot",
                }
            ],
            "review_configs": [
                {"allowed_decisions": ["approve", "edit", "reject"]}
            ],
        },
    )

    processor = _processor()
    event = {
        "type": "updates",
        "ns": (),
        "data": {"__interrupt__": (interrupt,)},
        "interrupts": (interrupt,),
    }
    payloads = [
        *await _collect(processor, event),
        *await _collect(processor, event),
    ]

    assert payloads == [
        {
            "type": "approval_required",
            "interrupt_id": "interrupt-1",
            "action_name": "execute",
            "command": "python plot.py",
            "description": "Create a plot",
            "limits": {
                "timeout_seconds": 600,
                "network": "none",
                "cpu": "2 vCPU",
                "memory": "4 GB",
            },
            "allowed_decisions": ["approve", "edit", "reject"],
        }
    ]


@pytest.mark.asyncio
async def test_specialist_question_becomes_a_clarification_event() -> None:
    """A question raised inside a subagent still reaches the user's stream."""

    interrupt = Interrupt(
        id="interrupt-2",
        value={
            "kind": "clarification",
            "asked_by": "software-specialist",
            "question": "Which background subtraction should I use?",
            "options": ["Solvent-only", "Empty cell", 7],
        },
    )

    payloads = await _collect(
        _processor(),
        {
            "type": "updates",
            "ns": ("tools:abc",),
            "data": {"__interrupt__": (interrupt,)},
        },
    )

    assert payloads == [
        {
            "type": "clarification_required",
            "interrupt_id": "interrupt-2",
            "asked_by": "software-specialist",
            "question": "Which background subtraction should I use?",
            "options": ["Solvent-only", "Empty cell"],
        }
    ]


@pytest.mark.asyncio
async def test_an_unrecognisable_interrupt_is_not_streamed() -> None:
    payloads = await _collect(
        _processor(),
        {
            "type": "updates",
            "ns": (),
            "data": {"__interrupt__": (Interrupt(id="interrupt-3", value={"kind": "other"}),)},
        },
    )

    assert payloads == []


@pytest.mark.asyncio
async def test_resumed_interrupt_is_not_emitted_as_a_stale_approval() -> None:
    interrupt = Interrupt(
        id="interrupt-1",
        value={
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "python plot.py"},
                    "description": "Create a plot",
                }
            ],
            "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
        },
    )

    payloads = await _collect(
        _processor(ignored_interrupt_ids={"interrupt-1"}),
        {
            "type": "updates",
            "ns": (),
            "data": {"__interrupt__": (interrupt,)},
        },
    )

    assert payloads == []


@pytest.mark.asyncio
async def test_v2_custom_stream_accepts_only_sandbox_status() -> None:
    accepted = await _collect(
        _processor(),
        {
            "type": "custom",
            "ns": ("tools:abc",),
            "data": {
                "type": "sandbox_status",
                "status": "queued",
                "label": "Sandbox command queued",
            },
        },
    )
    ignored = await _collect(
        _processor(),
        {"type": "custom", "ns": (), "data": {"secret": "must not leak"}},
    )

    assert accepted == [
        {
            "type": "sandbox_status",
            "status": "queued",
            "label": "Sandbox command queued",
        }
    ]
    assert ignored == []


@pytest.mark.asyncio
async def test_tool_results_become_status_not_transcript_payloads() -> None:
    """Tool payloads are large and the model has already read them; rendering
    them in the transcript buries the answer."""
    tool_message = ToolMessage(
        content="x" * 5000, tool_call_id="call-1", name="search_code_semantic", id="t-1"
    )
    payloads = await _collect(
        _processor(), (("model",), "updates", {"tools": {"messages": [tool_message]}})
    )

    assert [p["type"] for p in payloads] == ["status"]
    assert payloads[0]["phase"] == events.PHASE_TOOL
    assert payloads[0]["tool"] == "search_code_semantic"
    assert len(payloads[0]["summary"]) < 200


@pytest.mark.asyncio
async def test_tool_payloads_can_be_enabled_for_debugging() -> None:
    tool_message = ToolMessage(content="result", tool_call_id="c1", name="tool", id="t-1")
    payloads = await _collect(
        _processor(include_tool_payloads=True),
        (("model",), "updates", {"tools": {"messages": [tool_message]}}),
    )

    assert [p["type"] for p in payloads] == ["status", "message"]
    assert payloads[1]["content"]["content"] == "result"


@pytest.mark.asyncio
async def test_execute_tool_output_is_never_streamed_or_previewed() -> None:
    tool_message = ToolMessage(
        content="data:image/png;base64," + "x" * 5000,
        tool_call_id="c1",
        name="execute",
        id="t-1",
    )

    payloads = await _collect(
        _processor(include_tool_payloads=True),
        (("model",), "updates", {"tools": {"messages": [tool_message]}}),
    )

    assert payloads == []


# --------------------------------------------------------------------------
# message filtering
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_message_ids_are_sent_once() -> None:
    processor = _processor()
    event = (("model",), "updates", {"agent": {"messages": [AIMessage(content="hi", id="m-1")]}})

    first = await _collect(processor, event)
    second = await _collect(processor, event)

    assert [p["type"] for p in first] == ["message"]
    assert second == []


@pytest.mark.asyncio
async def test_user_echo_and_empty_tool_call_messages_are_dropped() -> None:
    echo = HumanMessage(content="hello", id="h-1")
    tool_call_only = AIMessage(
        content="",
        id="a-1",
        tool_calls=[{"name": "search", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    payloads = await _collect(
        _processor(), (("model",), "updates", {"agent": {"messages": [echo, tool_call_only]}})
    )

    assert payloads == []


@pytest.mark.asyncio
async def test_subagent_final_packet_is_not_shown_as_an_answer() -> None:
    """The subagent returns a research packet for the supervisor to synthesize,
    not a user-facing answer."""
    payloads = await _collect(
        _processor(),
        (("tools:abc",), "updates", {"agent": {"messages": [AIMessage(content="packet", id="s-1")]}}),
    )

    assert all(p["type"] != "message" for p in payloads)


# --------------------------------------------------------------------------
# the `tools:` namespace convention, checked against a real graph
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subagent_detection_matches_real_langgraph_namespaces() -> None:
    """`is_subagent_namespace` encodes a LangGraph/deepagents convention rather
    than a documented API. Pin it against an actual run so an upgrade that
    changes the namespace shape fails here instead of silently streaming
    subagent reasoning to users."""
    from deepagents.backends import StateBackend
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    class _Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    subagent = create_agent(
        model=_Model(responses=[AIMessage(content="research packet")]),
        tools=[],
        name="researcher",
    )
    supervisor = create_agent(
        model=_Model(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "task",
                        "args": {"description": "go", "subagent_type": "researcher"},
                        "id": "c1",
                        "type": "tool_call",
                    }],
                ),
                AIMessage(content="final answer"),
            ]
        ),
        tools=[],
        middleware=[
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[{
                    "name": "researcher",
                    "description": "d",
                    "runnable": subagent,
                }],
            )
        ],
        checkpointer=InMemorySaver(),
    )

    namespaces = set()
    async for node_path, _mode, _event in supervisor.astream(
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "t"}},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        namespaces.add(tuple(node_path))

    classified = {ns: is_subagent_namespace(ns) for ns in namespaces}
    assert any(classified.values()), f"no subagent namespace detected in {namespaces}"
    assert any(not flag for flag in classified.values()), "supervisor traffic missing"
