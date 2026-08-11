"""Validation and LangGraph resume commands for pending interrupts.

Two kinds travel through this pipe: a sandbox `execute` approval raised by
`HumanInTheLoopMiddleware`, and a clarification question raised by the `ask_user`
tool. Both are resumed by interrupt id, so a question asked from inside a
specialist subgraph resumes that specialist exactly where it paused.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from juena.agents.ask_user import CLARIFICATION_KIND
from juena.sandbox.artifacts import get_artifact_store
from juena.sandbox.schemas import (
    ApprovalResumeInput,
    ClarificationResumeInput,
    ResumeInput,
)

EXECUTE_APPROVAL_KIND = "execute_approval"


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


def interrupt_kind(interrupt: Interrupt) -> str | None:
    """Classify one pending interrupt, or return None when it is unusable."""

    value = interrupt.value
    if not isinstance(value, dict):
        return None
    if value.get("kind") == CLARIFICATION_KIND:
        return CLARIFICATION_KIND if isinstance(value.get("question"), str) else None
    try:
        _execute_action(interrupt)
    except ApprovalError:
        return None
    return EXECUTE_APPROVAL_KIND


async def get_first_pending_interrupt(
    agent: CompiledStateGraph,
    config: dict[str, Any],
) -> tuple[Interrupt, str] | None:
    """Return the first usable pending interrupt, including after a UI reload."""

    snapshot = await agent.aget_state(config=config)
    for task in getattr(snapshot, "tasks", ()):
        for interrupt in getattr(task, "interrupts", ()):
            if not isinstance(interrupt, Interrupt):
                continue
            kind = interrupt_kind(interrupt)
            if kind is not None:
                return interrupt, kind
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


def _execute_resume(
    *,
    interrupt: Interrupt,
    user_id: str,
    payload: ApprovalResumeInput,
) -> Command:
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

    get_artifact_store().audit(
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
    return Command(resume={payload.interrupt_id: {"decisions": [decision]}})


def _clarification_resume(
    *,
    interrupt: Interrupt,
    user_id: str,
    payload: ClarificationResumeInput,
) -> Command:
    if interrupt_kind(interrupt) != CLARIFICATION_KIND:
        raise ApprovalError("This interrupt is not a pending question")
    answer = payload.answer.strip()
    if not answer:
        raise ApprovalError("An answer must not be empty")
    get_artifact_store().audit(
        {
            "event": "clarification_answered",
            "user_id": user_id,
            "thread_id": payload.thread_id,
            "interrupt_id": payload.interrupt_id,
        }
    )
    # The tool returns this string, so the asking agent resumes with the user's
    # own words rather than a decision code.
    return Command(resume={payload.interrupt_id: answer})


async def build_resume_command(
    *,
    agent: CompiledStateGraph,
    config: dict[str, Any],
    user_id: str,
    payload: ResumeInput,
) -> Command:
    """Validate one pending interrupt and build its typed resume command."""

    interrupt = await get_pending_interrupt(agent, config, payload.interrupt_id)
    if isinstance(payload, ClarificationResumeInput):
        return _clarification_resume(interrupt=interrupt, user_id=user_id, payload=payload)
    if interrupt_kind(interrupt) != EXECUTE_APPROVAL_KIND:
        raise ApprovalError("This interrupt is not a pending sandbox command")
    return _execute_resume(interrupt=interrupt, user_id=user_id, payload=payload)
