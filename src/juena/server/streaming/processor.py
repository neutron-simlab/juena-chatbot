"""Turn LangGraph stream events into the SSE events the UI consumes.

Two rules shape this module:

1. Only the coordinator's tokens are the answer. A subagent's tokens are its
   internal reasoning; streaming them makes the UI show text that is later
   replaced by the real answer.
2. Tool results are activity, not transcript. They are large, the model has
   already read them, and rendering them buries the answer. They become compact
   `status` events unless payloads are explicitly enabled.
"""

from typing import Any, AsyncGenerator

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from juena.core.log import get_logger
from juena.server.errors import MessageProcessingError, StreamingError
from juena.server.streaming import events
from juena.server.streaming.handlers import (
    extract_update_messages,
    is_subagent_namespace,
    token_text,
)
from juena.server.utils import langchain_to_chat_message

logger = get_logger(__name__)

# Preview length for a tool result shown inside the status container.
_TOOL_SUMMARY_CHARS = 120


def _tool_summary(message: ToolMessage) -> str:
    """One-line gist of a tool result, for the activity log."""
    text = " ".join(message.text.split())
    if len(text) <= _TOOL_SUMMARY_CHARS:
        return text
    return text[:_TOOL_SUMMARY_CHARS].rstrip() + "…"


class StreamEventProcessor:
    """Processor for LangGraph stream events."""

    def __init__(
        self,
        agent: Any,
        config: Any,
        run_id: str,
        user_input_message: str,
        *,
        include_tool_payloads: bool = False,
    ):
        self.agent = agent
        self.config = config
        self.run_id = run_id
        self.user_input_message = user_input_message
        self.include_tool_payloads = include_tool_payloads

        # Messages already sent this run, keyed by LangChain message id.
        # `updates` only emits new messages -- including on a resumed thread --
        # so this guards against the same message arriving twice within one run,
        # not against replayed history.
        self._streamed_message_ids: set[str] = set()

        # Tracks the last status emitted so identical ones are not repeated.
        self._last_status: tuple[str, str] | None = None

    def _parse_stream_event(self, stream_event: Any) -> tuple[str, Any, Any]:
        """Split a raw stream event into (stream_mode, event, node_path)."""
        if not isinstance(stream_event, tuple):
            raise StreamingError(f"Unexpected stream event type: {type(stream_event)}")

        if len(stream_event) == 3:
            # subgraphs=True: (node_path, stream_mode, event)
            node_path, stream_mode, event = stream_event
            return stream_mode, event, node_path

        stream_mode, event = stream_event
        return stream_mode, event, None

    def _status(self, phase: str, label: str, **extra: Any) -> dict[str, Any] | None:
        """Build a status event, or None when it repeats the previous one."""
        key = (phase, label)
        if key == self._last_status:
            return None
        self._last_status = key
        return events.status_event(phase, label, **extra)

    async def process_event(self, stream_event: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Process one stream event, yielding SSE payload dicts."""
        try:
            stream_mode, event, node_path = self._parse_stream_event(stream_event)
            from_subagent = is_subagent_namespace(node_path)

            if stream_mode == "messages":
                if from_subagent:
                    # Subagent reasoning is activity, not the answer.
                    status = self._status(
                        events.PHASE_RESEARCHING, "Researching the repository…"
                    )
                    if status:
                        yield status
                    return

                text = token_text(event)
                if text:
                    status = self._status(events.PHASE_DONE, "")
                    if status:
                        yield status
                    yield events.token_event(text)
                return

            if stream_mode == "updates":
                for message in extract_update_messages(event):
                    async for payload in self._handle_message(message, from_subagent):
                        yield payload

        except Exception as e:
            logger.error(f"Error processing stream event: {e}", exc_info=True)
            error = StreamingError("Failed to process stream event", details={"error": str(e)})
            yield events.error_event(error.message)

    async def _handle_message(
        self, message: BaseMessage, from_subagent: bool
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert one complete message into the events the client should see."""
        try:
            if isinstance(message, SystemMessage):
                return

            if isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", None) or "tool"
                yield events.status_event(
                    events.PHASE_TOOL,
                    f"Ran {tool_name}",
                    tool=tool_name,
                    summary=_tool_summary(message),
                )
                if not self.include_tool_payloads:
                    return

            chat_message = langchain_to_chat_message(message)
            chat_message.run_id = str(self.run_id)

            # A subagent's final packet is input to the coordinator, not an
            # answer for the user; the coordinator synthesizes from it.
            if from_subagent and chat_message.type == "ai":
                return

            # Echo of the user's own turn.
            if chat_message.type == "human" and chat_message.content == self.user_input_message:
                return
            if chat_message.type == "system":
                return

            # Tool-call announcements carry no content of their own.
            if (
                chat_message.type == "ai"
                and chat_message.tool_calls
                and not (chat_message.content or "").strip()
            ):
                return

            message_id = chat_message.id
            if message_id:
                if message_id in self._streamed_message_ids:
                    return
                self._streamed_message_ids.add(message_id)

            yield events.message_event(chat_message)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error = MessageProcessingError(
                "Failed to process message",
                message_type=type(message).__name__,
                details={"error": str(e)},
            )
            yield events.error_event(error.message)
