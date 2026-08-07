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
