"""Agent middleware for artifact hand-off and sandbox execution records."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from juena.sandbox.artifacts import ARTIFACT_MESSAGE_KEY, get_artifact_store
from juena.sandbox.policy import requires_execution_approval
from juena.sandbox.runtime import _context_value


class ArtifactMessageMiddleware(AgentMiddleware):
    """Attach pending artifact references to the supervisor's final AI message."""

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last_ai = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
        if last_ai is None or last_ai.tool_calls:
            return None
        context = getattr(runtime, "context", None)
        user_id = _context_value(context, "user_id")
        thread_id = _context_value(context, "thread_id")
        if not user_id or not thread_id:
            return None
        artifacts = get_artifact_store().claim_for_message(user_id, thread_id)
        if not artifacts:
            return None
        last_ai.additional_kwargs = {
            **last_ai.additional_kwargs,
            ARTIFACT_MESSAGE_KEY: artifacts,
        }
        return {"messages": [last_ai]}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def _identity(runtime: Any) -> tuple[str, str, str | None] | None:
    """Return (user_id, thread_id, run_id) when the caller is authenticated."""
    context = getattr(runtime, "context", None)
    user_id = _context_value(context, "user_id")
    thread_id = _context_value(context, "thread_id")
    if not user_id or not thread_id:
        return None
    config = getattr(runtime, "config", None)
    run_id = config.get("run_id") if hasattr(config, "get") else None
    return user_id, thread_id, str(run_id) if run_id else None


def _result_text(result: Any) -> str:
    """Pull the tool's reply text out of a ToolMessage or Command."""
    if isinstance(result, ToolMessage):
        return result.text
    update = getattr(result, "update", None)
    messages = update.get("messages", []) if isinstance(update, dict) else []
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, ToolMessage):
            return message.text
    return ""


class SandboxExecutionMiddleware(AgentMiddleware):
    """Record the approved sandbox command and its output as artifacts.

    `HumanInTheLoopMiddleware` raises its approval interrupt in `after_model`,
    which runs before tools execute. Reaching `wrap_tool_call` therefore means
    the human has already approved -- and `request.tool_call` carries the final,
    post-edit command. No approval state has to be stored or matched by text.
    """

    def _before(self, request: Any) -> tuple[str, str, str | None, str] | None:
        """Persist the command about to run; return the identity to reuse."""
        tool_call = request.tool_call
        if not isinstance(tool_call, dict) or tool_call.get("name") != "execute":
            return None
        if not requires_execution_approval(request):
            return None
        identity = _identity(getattr(request, "runtime", None))
        if identity is None:
            return None
        args = tool_call.get("args")
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str) or not command.strip():
            return None
        user_id, thread_id, run_id = identity
        store = get_artifact_store()
        try:
            store.register_artifact(
                user_id=user_id,
                thread_id=thread_id,
                run_id=run_id,
                filename="generated-command.sh",
                content=(command.rstrip() + "\n").encode("utf-8"),
                caption="Generated command",
            )
        except (ValueError, OSError):
            pass
        return user_id, thread_id, run_id, command

    @staticmethod
    def _after(record: tuple[str, str, str | None, str], result: Any) -> None:
        user_id, thread_id, run_id, command = record
        text = _result_text(result)
        store = get_artifact_store()
        if text:
            try:
                store.register_artifact(
                    user_id=user_id,
                    thread_id=thread_id,
                    run_id=run_id,
                    filename="execution-output.txt",
                    content=text.encode("utf-8"),
                    caption="Execution output",
                )
            except (ValueError, OSError):
                pass
        store.audit(
            {
                "event": "sandbox_execution_approved",
                "user_id": user_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "command": command,
            }
        )

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        record = self._before(request)
        if record is None:
            return handler(request)
        result = handler(request)
        self._after(record, result)
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        record = self._before(request)
        if record is None:
            return await handler(request)
        result = await handler(request)
        self._after(record, result)
        return result

