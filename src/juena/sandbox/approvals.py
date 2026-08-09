"""Validation and LangGraph resume commands for sandbox approvals."""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from juena.sandbox.artifacts import get_artifact_store
from juena.sandbox.schemas import ApprovalResumeInput


class ApprovalError(ValueError):
    """Raised for stale, malformed, or policy-changing approval decisions."""


async def get_pending_interrupt(
    agent: CompiledStateGraph,
    config: dict[str, Any],
    interrupt_id: str,
) -> Interrupt:
    snapshot = await agent.aget_state(config=config)
    for task in getattr(snapshot, "tasks", ()):
        for interrupt in getattr(task, "interrupts", ()):
            if isinstance(interrupt, Interrupt) and interrupt.id == interrupt_id:
                return interrupt
    raise ApprovalError("Approval is stale, already used, or does not belong to this thread")


async def get_first_pending_execute_interrupt(
    agent: CompiledStateGraph,
    config: dict[str, Any],
) -> Interrupt | None:
    """Return the first valid execute interrupt, including after a UI reload."""
    snapshot = await agent.aget_state(config=config)
    for task in getattr(snapshot, "tasks", ()):
        for interrupt in getattr(task, "interrupts", ()):
            if not isinstance(interrupt, Interrupt):
                continue
            try:
                _execute_action(interrupt)
            except ApprovalError:
                continue
            return interrupt
    return None


def _execute_action(interrupt: Interrupt) -> dict[str, Any]:
    value = interrupt.value
    if not isinstance(value, dict):
        raise ApprovalError("Malformed approval request")
    actions = value.get("action_requests")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise ApprovalError("Expected exactly one sandbox action")
    action = actions[0]
    args = action.get("args")
    if action.get("name") != "execute" or not isinstance(args, dict):
        raise ApprovalError("Only sandbox execute approvals are accepted")
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ApprovalError("Sandbox command is missing")
    return action


async def build_resume_command(
    *,
    agent: CompiledStateGraph,
    config: dict[str, Any],
    user_id: str,
    payload: ApprovalResumeInput,
) -> Command:
    interrupt = await get_pending_interrupt(agent, config, payload.interrupt_id)
    action = _execute_action(interrupt)
    original_args = dict(action["args"])

    if payload.decision == "approve":
        if payload.edited_command is not None:
            raise ApprovalError("edited_command is valid only for an edit decision")
        decision: dict[str, Any] = {"type": "approve"}
        edited_command = None
    elif payload.decision == "reject":
        if payload.edited_command is not None:
            raise ApprovalError("edited_command is valid only for an edit decision")
        decision = {
            "type": "reject",
            "message": "The user rejected this sandbox command.",
        }
        edited_command = None
    else:
        edited_command = (payload.edited_command or "").strip()
        if not edited_command:
            raise ApprovalError("An edited command must not be empty")
        edited_args = {**original_args, "command": edited_command}
        decision = {
            "type": "edit",
            "edited_action": {
                "name": "execute",
                "args": edited_args,
            },
        }

    store = get_artifact_store()
    store.audit(
        {
            "event": "sandbox_approval",
            "user_id": user_id,
            "thread_id": payload.thread_id,
            "interrupt_id": payload.interrupt_id,
            "original_command": original_args["command"],
            "edited_command": edited_command,
            "decision": payload.decision,
        }
    )
    if payload.decision != "reject":
        store.mark_approved_command(
            user_id,
            payload.thread_id,
            payload.interrupt_id,
            edited_command or original_args["command"],
        )
    return Command(
        resume={
            payload.interrupt_id: {
                "decisions": [decision],
            }
        }
    )
