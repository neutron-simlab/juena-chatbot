"""Break deterministic tool-call loops before they exhaust the model budget."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage

#: Marks a refusal as ours, so counting them never depends on the wording. Prose
#: matching would break the moment the message is reworded or translated, and
#: would miscount a tool that happened to echo the text back.
GUARD_FLAG = "juena_repeated_tool_call"

REPEAT_ERROR = (
    "Identical call to `{name}` was already made and returned the same result. "
    "Repeating it cannot make progress. Take a different action, or finish and "
    "report what you have."
)

STOPPED = (
    "Stopped after {count} identical calls to `{name}`. Its result was already in "
    "hand, so repeating it could add nothing. Report what was established before "
    "this point."
)

#: Refusals tolerated before the run ends. The first is a nudge, and a model that
#: can act on it does. A second identical call is evidence the message is not
#: landing, and every round after that costs a model call and changes nothing --
#: one live run spent 59 of its 60 calls exactly this way.
REPEAT_LIMIT = 2


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


def _is_refusal(message: Any) -> bool:
    return (
        isinstance(message, ToolMessage)
        and bool(getattr(message, "additional_kwargs", {}).get(GUARD_FLAG))
    )


def _trailing_refusals(messages: list[Any]) -> tuple[int, str]:
    """How many refusals close the history unbroken, and the tool they were for.

    Walks back over the alternating ``AIMessage`` / refusal pairs a locked model
    produces, and stops at the first message that is neither -- so a single stray
    refusal earlier in a healthy run never accumulates.
    """
    count = 0
    name = "tool"
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            continue
        if _is_refusal(message):
            count += 1
            name = message.name or name
            continue
        break
    return count, name


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Refuse a tool call identical to the one that just returned, then stop.

    At `temperature=0.0` a tool whose result adds no new information leaves the
    next context effectively unchanged, so a model can lock into re-issuing it.
    The refusal alone does not break that: it is a sentence handed back to the
    model that just ignored the tool result, and a model that cannot act on the
    result often cannot act on the sentence either.

    So the refusal buys the cheap win -- the tool does not run, which matters most
    for `execute` -- and `before_model` ends the run once the refusals stack up.
    `SpecialistOutcomeMiddleware` is the specialist's exit node, so ending here
    still produces a report saying the run was cut short.
    """

    def __init__(self, *, repeat_limit: int = REPEAT_LIMIT) -> None:
        super().__init__()
        self._repeat_limit = repeat_limit

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
            additional_kwargs={GUARD_FLAG: True},
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

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        count, name = _trailing_refusals(list(messages))
        if count < self._repeat_limit:
            return None
        return {
            "jump_to": "end",
            # The count includes the call that succeeded, which is what the model
            # actually issued that many times.
            "messages": [AIMessage(content=STOPPED.format(count=count + 1, name=name))],
        }

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
