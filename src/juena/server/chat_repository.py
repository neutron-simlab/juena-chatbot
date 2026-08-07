"""User-owned chat persistence operations.

Ownership metadata lives in the ``chats`` table.  Message history is *not*
duplicated here: the LangGraph checkpointer already stores every message for a
thread, so :func:`load_thread_messages` reads it back from there.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena.core.log import get_logger
from juena.schema.server import ChatMessage
from juena.server.checkpointer import get_checkpointer
from juena.server.models import Chat
from juena.server.utils import langchain_to_chat_message

logger = get_logger(__name__)


class ChatNotFoundError(LookupError):
    """Raised for missing chats and chats owned by another user."""


async def ensure_owned_chat(
    session: AsyncSession,
    user_id: UUID,
    thread_id: str | None,
) -> Chat:
    """Create a chat or verify ownership without revealing other users' rows."""

    effective_thread_id = thread_id or str(uuid4())
    chat = await session.get(Chat, effective_thread_id)
    if chat is None:
        chat = Chat(thread_id=effective_thread_id, user_id=user_id)
        session.add(chat)
        await session.flush()
    elif chat.user_id != user_id:
        raise ChatNotFoundError(effective_thread_id)
    return chat


async def get_owned_chat(session: AsyncSession, user_id: UUID, thread_id: str) -> Chat:
    result = await session.execute(
        select(Chat).where(Chat.thread_id == thread_id, Chat.user_id == user_id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise ChatNotFoundError(thread_id)
    return chat


async def list_owned_chats(
    session: AsyncSession,
    user_id: UUID,
    limit: int = 50,
) -> list[Chat]:
    result = await session.execute(
        select(Chat)
        .where(Chat.user_id == user_id)
        .order_by(Chat.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars())


async def load_thread_messages(thread_id: str) -> list[ChatMessage]:
    """Read a thread's message history from the LangGraph checkpointer.

    The checkpointer is the single source of truth for conversation content.
    Callers must verify ownership of ``thread_id`` before calling this.
    """

    checkpoint = await get_checkpointer().aget(
        {"configurable": {"thread_id": thread_id}}
    )
    if checkpoint is None:
        return []

    messages: list[ChatMessage] = []
    for message in checkpoint.get("channel_values", {}).get("messages", []):
        if not isinstance(message, BaseMessage) or isinstance(message, SystemMessage):
            continue
        # Tool results are activity, not transcript. The live stream reports
        # them as compact status events rather than messages, so including them
        # here would make a reloaded thread render differently from how the
        # user watched it happen.
        if isinstance(message, ToolMessage) and not global_config.STREAM_TOOL_PAYLOADS:
            continue
        try:
            chat_message = langchain_to_chat_message(message)
        except ValueError:
            logger.warning("Skipping unsupported checkpointed message in thread %s", thread_id)
            continue
        # Tool-call announcements have no content of their own; the live stream
        # drops them too.
        if (
            chat_message.type == "ai"
            and chat_message.tool_calls
            and not (chat_message.content or "").strip()
        ):
            continue
        chat_message.thread_id = thread_id
        messages.append(chat_message)
    return messages


def chat_to_dict(chat: Chat) -> dict[str, str]:
    return {
        "thread_id": chat.thread_id,
        "title": chat.title,
        "summary": chat.summary,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }
