from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from juena.schema.server import ChatMessage
from juena.server.chat.input_constants import (
    DISPLAY_ATTACHMENTS_KEY,
    DISPLAY_TEXT_KEY,
)
from juena.sandbox.artifacts import ARTIFACT_MESSAGE_KEY

# Content-block types that carry the model's internal reasoning rather than the
# answer. Surfaced separately so the UI can collapse them instead of splicing
# them into the response text.
_REASONING_BLOCK_TYPES = frozenset({"reasoning", "thinking"})


def message_text(message: BaseMessage) -> str:
    """Return the human-readable text of a message.

    ``BaseMessage.text`` concatenates the text blocks of any content shape --
    plain string, list of strings, or list of typed blocks -- and ignores
    non-text blocks such as reasoning or images.
    """
    return message.text


def reasoning_from(message: BaseMessage) -> str:
    """Return concatenated reasoning/thinking content, or "" when there is none.

    Uses ``content_blocks``, which normalises each provider's native reasoning
    representation into a common ``{"type": "reasoning", "reasoning": ...}``
    shape.
    """
    try:
        blocks = message.content_blocks
    except Exception:  # noqa: BLE001 - never fail a response over reasoning text
        return ""

    parts = [
        str(block.get("reasoning") or block.get("thinking") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") in _REASONING_BLOCK_TYPES
    ]
    return "".join(part for part in parts if part)


def display_text_from(message: BaseMessage) -> str:
    """Return the text to show for *message* in the transcript.

    A human turn that staged uploads or pasted code carries the ``/inputs``
    manifest as its content, because that is what the model must read. The text
    the user actually typed is preserved alongside it, so prefer that when it is
    present -- otherwise a reloaded chat renders agent scaffolding in the user's
    own bubble.
    """
    original = message.additional_kwargs.get(DISPLAY_TEXT_KEY)
    if isinstance(original, str) and original.strip():
        return original
    return message_text(message)


def display_attachments_from(message: BaseMessage) -> list[dict[str, object]]:
    """Return the upload metadata recorded on *message*, or [] when there is none."""
    attachments = message.additional_kwargs.get(DISPLAY_ATTACHMENTS_KEY)
    if not isinstance(attachments, list):
        return []
    return [item for item in attachments if isinstance(item, dict)]


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message.

    Args:
        message: LangChain message to convert
    """
    message_id = getattr(message, "id", None)

    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                id=message_id,
                content=display_text_from(message),
            )
            if attachments := display_attachments_from(message):
                human_message.custom_data["attachments"] = attachments
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                id=message_id,
                content=message_text(message),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            if reasoning := reasoning_from(message):
                ai_message.custom_data["reasoning"] = reasoning
            artifacts = message.additional_kwargs.get(ARTIFACT_MESSAGE_KEY)
            if isinstance(artifacts, list):
                ai_message.custom_data["artifacts"] = [
                    item for item in artifacts if isinstance(item, dict)
                ]
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                id=message_id,
                content=message_text(message),
                tool_call_id=message.tool_call_id,
            )
            tool_message.custom_data = {
                "tool_kind": "regular_tool_result",
                "display_mode": "collapsed_by_default",
            }
            if tool_name := getattr(message, "name", None):
                tool_message.custom_data["tool_name"] = tool_name
            return tool_message
        case SystemMessage():
            return ChatMessage(
                type="system",
                id=message_id,
                content=message_text(message),
            )
        case LangchainChatMessage():
            if message.role == "custom":
                # Custom messages carry a single dict payload. A str content is
                # a caller error: indexing it would silently yield one char.
                payload = message.content
                if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                    custom_data = payload[0]
                elif isinstance(payload, dict):
                    custom_data = payload
                else:
                    raise ValueError(
                        "Custom chat message content must be a dict or a list of dicts, "
                        f"got {type(payload).__name__}"
                    )
                return ChatMessage(
                    type="custom",
                    id=message_id,
                    content="",
                    custom_data=custom_data,
                )
            raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
