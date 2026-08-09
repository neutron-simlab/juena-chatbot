"""Handlers for the LangGraph stream modes this server subscribes to.

Two modes are consumed:

``updates``   complete messages as each node finishes
``messages``  token chunks from LLM calls

``custom`` is deliberately not subscribed: nothing in this codebase calls
``get_stream_writer()``, and the old handler passed raw payloads into the
message converter, which rejected them.
"""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.types import Interrupt, Overwrite

# With subgraphs=True the namespace of a subagent contains a `tools:`-prefixed
# segment, because subagents are invoked through the parent's tool node.
# Verified against a real run: the supervisor streams under `()` and a subagent
# under `('tools:<task-uuid>',)`. The suffix is a task id, not a readable name,
# so there is nothing here worth showing the user.
_SUBAGENT_NAMESPACE_PREFIX = "tools:"


def is_subagent_namespace(node_path: Any) -> bool:
    """Whether a stream event originated inside a subagent rather than the supervisor."""
    if not node_path:
        return False
    return any(str(segment).startswith(_SUBAGENT_NAMESPACE_PREFIX) for segment in node_path)


def extract_update_messages(event: dict[str, Any]) -> list[BaseMessage]:
    """Pull the messages out of an `updates` event payload."""
    new_messages: list[BaseMessage] = []

    for node, updates in (event or {}).items():
        if node == "__interrupt__":
            interrupt: Interrupt
            for interrupt in updates:
                new_messages.append(AIMessage(content=interrupt.value))
            continue

        update_messages = (updates or {}).get("messages", [])
        if isinstance(update_messages, Overwrite):
            update_messages = update_messages.value
        if update_messages is None:
            continue
        if isinstance(update_messages, BaseMessage):
            new_messages.append(update_messages)
            continue
        if isinstance(update_messages, list):
            new_messages.extend(m for m in update_messages if isinstance(m, BaseMessage))

    return new_messages


def token_text(event: tuple[BaseMessage, dict[str, Any]]) -> str | None:
    """Return the streamable text of a `messages` event, or None to skip it.

    Only ``AIMessageChunk`` carries answer text. ``.text`` yields the text
    blocks alone, so tool-call and reasoning blocks never reach the client.
    """
    msg, metadata = event

    if "skip_stream" in (metadata or {}).get("tags", []):
        return None
    if not isinstance(msg, AIMessageChunk):
        return None

    return msg.text or None
