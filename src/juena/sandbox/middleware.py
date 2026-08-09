"""Agent middleware that associates generated artifacts with final answers."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from juena.sandbox.artifacts import ARTIFACT_MESSAGE_KEY, get_artifact_store
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

