"""Agent middleware for artifact hand-off and sandbox execution records."""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from juena.sandbox.artifacts import (
    ARTIFACT_MESSAGE_KEY,
    get_artifact_store,
)
from juena.sandbox.evidence import capture_sandbox_execution
from juena.schema.sandbox import SandboxExecutionEvidence
from juena.sandbox.policy import requires_execution_approval
from juena.sandbox.runtime import _context_value


def _final_ai_message(state: Any) -> AIMessage | None:
    """Return the completed model response, excluding tool-call requests."""
    messages = state.get("messages", []) if isinstance(state, dict) else []
    last_ai = next((item for item in reversed(messages) if isinstance(item, AIMessage)), None)
    return None if last_ai is None or last_ai.tool_calls else last_ai


class ArtifactMessageMiddleware(AgentMiddleware):
    """Attach pending artifact references to the supervisor's final AI message.

    On `after_agent` rather than `after_model`, because a budget that ends a run
    returns `jump_to: "end"` and the graph routes straight past every
    `after_model` hook. A supervisor turn cut short that way used to drop its
    pending artifacts silently -- the plots were generated, collected, and then
    never attached to anything the user could see.

    `after_agent` runs once, on every path out of the graph, which is exactly the
    guarantee this needs. It costs nothing on the ordinary path: the claim is a
    no-op until the final model response has no tool calls left to make.
    """

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last_ai = _final_ai_message(state)
        if last_ai is None:
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

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


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


def _tool_message(result: Any) -> ToolMessage | None:
    """Find the ToolMessage inside a raw result or a Command update."""
    if isinstance(result, ToolMessage):
        return result
    update = getattr(result, "update", None)
    messages = update.get("messages", []) if isinstance(update, dict) else []
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, ToolMessage):
            return message
    return None


def _result_text(result: Any) -> str:
    """Pull the tool's reply text out of a ToolMessage or Command."""
    message = _tool_message(result)
    return message.text if message is not None else ""


class SandboxExecutionState(AgentState):
    """Specialist-private direct execution evidence."""

    sandbox_execution_events: NotRequired[
        Annotated[list[dict[str, Any]], operator.add, PrivateStateAttr]
    ]


class SandboxExecutionMiddleware(AgentMiddleware[SandboxExecutionState]):
    """Capture typed execution evidence, and record approved commands as artifacts.

    `HumanInTheLoopMiddleware` raises its approval interrupt in `after_model`,
    which runs before tools execute. Reaching `wrap_tool_call` therefore means
    the human has already approved -- and `request.tool_call` carries the final,
    post-edit command. No approval state has to be stored or matched by text.

    The two jobs are deliberately kept apart. Evidence is what later proves a
    command ran and what it exited with, so it is captured for every `execute`
    call; only the audit bookkeeping needs an authenticated identity. Gating
    both on the identity would turn a missing user id into a silently
    fail-open run.
    """

    @staticmethod
    def _command(request: Any) -> str | None:
        """The command this call will run, or None when it is not an execute."""
        tool_call = request.tool_call
        if not isinstance(tool_call, dict) or tool_call.get("name") != "execute":
            return None
        args = tool_call.get("args")
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str) or not command.strip():
            return None
        return command

    def _before(self, request: Any, command: str) -> tuple[str, str, str | None, str] | None:
        """Persist the command about to run; return the identity to reuse."""
        if not requires_execution_approval(request):
            return None
        identity = _identity(getattr(request, "runtime", None))
        if identity is None:
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
                category="record",
            )
        except (ValueError, OSError):
            pass
        return user_id, thread_id, run_id, command

    @classmethod
    def _after(cls, record: tuple[str, str, str | None, str], result: Any) -> None:
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
                    category="record",
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

    @staticmethod
    def _with_evidence(
        result: Any,
        command: str,
        events: list[SandboxExecutionEvidence],
    ) -> Command:
        message = _tool_message(result)
        if not events:
            events = [
                SandboxExecutionEvidence(
                    command=command,
                    status="tool_error",
                    exit_code=None,
                )
            ]
        last = events[-1]
        # A refused export never ran, so its guidance is not a tool failure.
        if message is not None and last.attempted:
            message.status = "success" if last.succeeded else "error"
        update = getattr(result, "update", None)
        merged = dict(update) if isinstance(update, dict) else {}
        if isinstance(result, ToolMessage):
            merged["messages"] = [result]
        merged["sandbox_execution_events"] = [
            event.model_dump(mode="json") for event in events
        ]
        if isinstance(result, Command):
            return replace(result, update=merged)
        return Command(update=merged)

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        command = self._command(request)
        if command is None:
            return handler(request)
        record = self._before(request, command)
        with capture_sandbox_execution() as events:
            result = handler(request)
        if record is not None:
            self._after(record, result)
        return self._with_evidence(result, command, events)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        command = self._command(request)
        if command is None:
            return await handler(request)
        record = self._before(request, command)
        with capture_sandbox_execution() as events:
            result = await handler(request)
        if record is not None:
            self._after(record, result)
        return self._with_evidence(result, command, events)
