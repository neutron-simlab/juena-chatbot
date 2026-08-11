"""Break deterministic tool-call loops before they exhaust the model budget."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

REPEAT_ERROR = (
    "Identical call to `{name}` was already made and returned the same result. "
    "Repeating it cannot make progress. Take a different action, or finish and "
    "report what you have."
)


def _signature(tool_call: dict[str, Any]) -> str:
    """Stable key for one tool call, insensitive to argument ordering."""

    args = tool_call.get("args")
    payload = args if isinstance(args, dict) else {}
    return json.dumps([tool_call.get("name"), payload], sort_keys=True, default=str)


def _previous_signature(messages: list[Any]) -> str | None:
    """Signature of the most recent tool call that already produced a result."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ToolMessage):
            continue
        for candidate in reversed(messages[:index]):
            if isinstance(candidate, AIMessage) and candidate.tool_calls:
                for call in candidate.tool_calls:
                    if call.get("id") == message.tool_call_id:
                        return _signature(call)
                break
        return None
    return None


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Refuse a tool call byte-identical to the one that just returned.

    At `temperature=0.0` a tool whose result adds no new information leaves the
    next context effectively unchanged, so a model can lock into re-issuing it
    until `ModelCallLimitMiddleware` ends the run. That silent exhaustion is what
    strands a specialist without a structured report, so the loop is cut here
    rather than absorbed downstream. The check reads the message history instead
    of private state so it survives summarization and approval resumes.
    """

    def _blocked(self, request: Any) -> ToolMessage | None:
        tool_call = request.tool_call
        if not isinstance(tool_call, dict):
            return None
        messages = request.state.get("messages", []) if hasattr(request, "state") else []
        if _previous_signature(list(messages)) != _signature(tool_call):
            return None
        name = tool_call.get("name") or "tool"
        return ToolMessage(
            content=REPEAT_ERROR.format(name=name),
            name=name,
            tool_call_id=tool_call.get("id") or "repeated-tool-call",
            status="error",
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        blocked = self._blocked(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        blocked = self._blocked(request)
        return blocked if blocked is not None else await handler(request)
