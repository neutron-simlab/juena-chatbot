"""SSE payload builders for the agent stream.

One place that knows the wire shape, so the frame format is not spelled out as
``f"data: {json.dumps(...)}"`` at a dozen call sites.

Event types on the wire:

``thread``   thread id, emitted once before any content
``token``    an incremental text chunk of the assistant's answer
``message``  a complete message (`ChatMessage` payload)
``status``   agent activity: thinking / researching / running a tool
``error``    a failure the client should surface
"""

from typing import Any

from juena.schema.server import ChatMessage

# Phases a client can render. `researching` means a subagent is working.
PHASE_THINKING = "thinking"
PHASE_RESEARCHING = "researching"
PHASE_TOOL = "tool"
PHASE_DONE = "done"


def thread_event(thread_id: str) -> dict[str, Any]:
    return {"type": "thread", "thread_id": thread_id}


def token_event(content: str) -> dict[str, Any]:
    return {"type": "token", "content": content}


def message_event(message: ChatMessage) -> dict[str, Any]:
    return {"type": "message", "content": message.model_dump(mode="json")}


def status_event(phase: str, label: str, **extra: Any) -> dict[str, Any]:
    return {"type": "status", "phase": phase, "label": label, **extra}


def error_event(content: str) -> dict[str, Any]:
    return {"type": "error", "content": content}


def approval_required_event(interrupt_id: str, value: dict[str, Any]) -> dict[str, Any]:
    action_requests = value.get("action_requests") or []
    review_configs = value.get("review_configs") or []
    action = action_requests[0] if action_requests and isinstance(action_requests[0], dict) else {}
    review = review_configs[0] if review_configs and isinstance(review_configs[0], dict) else {}
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    description = action.get("description")
    if isinstance(description, str) and "\n\nTool:" in description:
        description = description.split("\n\nTool:", 1)[0]
    return {
        "type": "approval_required",
        "interrupt_id": interrupt_id,
        "action_name": action.get("name"),
        "command": args.get("command"),
        "description": description,
        "limits": {
            "timeout_seconds": args.get("timeout"),
            "network": "none",
            "cpu": "2 vCPU",
            "memory": "4 GB",
        },
        "allowed_decisions": review.get("allowed_decisions") or [],
    }
