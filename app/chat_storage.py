"""Authenticated chat-history adapter for the Streamlit UI.

Postgres persistence is owned by the FastAPI service.  Streamlit keeps no local
conversation database and uses this small synchronous adapter to preserve the
existing sidebar interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from juena.clients.client import AgentClient
from juena.schema.server import ChatMessage


@dataclass
class Chat:
    """Represents a user-owned chat session."""

    thread_id: str
    title: str = "New Chat"
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def _parse_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return fallback or datetime.now()


def _chat_from_payload(payload: dict[str, Any]) -> Chat:
    return Chat(
        thread_id=str(payload["thread_id"]),
        title=str(payload.get("title") or "New Chat"),
        summary=str(payload.get("summary") or ""),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


class ChatStorage:
    """Remote storage facade backed by authenticated FastAPI chat endpoints."""

    def __init__(self, client: AgentClient) -> None:
        self.client = client

    def upsert_chat(self, chat: Chat) -> None:
        # Metadata-only probe: the caller is deciding create-vs-update and has
        # no use for the thread's message history.
        existing = self.client.get_chat(chat.thread_id, include_messages=False)
        if existing is None:
            self.client.create_chat(chat.thread_id, chat.title)
            return
        self.client.update_chat(
            chat.thread_id,
            title=chat.title,
            summary=chat.summary,
        )

    def get_chat(self, thread_id: str) -> Chat | None:
        payload = self.client.get_chat(thread_id, include_messages=False)
        return _chat_from_payload(payload) if payload is not None else None

    def list_chats(self, limit: int = 50) -> list[Chat]:
        return [_chat_from_payload(payload) for payload in self.client.list_chats(limit)]

    def load_chat_with_messages(self, thread_id: str) -> tuple[Chat, list[ChatMessage]] | None:
        """Fetch metadata and history in the single request that carries both."""

        payload = self.client.get_chat(thread_id)
        if payload is None:
            return None
        messages = [
            ChatMessage.model_validate(message) for message in payload.get("messages", [])
        ]
        return _chat_from_payload(payload), messages

    def load_messages(self, thread_id: str) -> list[ChatMessage]:
        loaded = self.load_chat_with_messages(thread_id)
        return loaded[1] if loaded is not None else []

    def delete_chat(self, thread_id: str) -> None:
        self.client.delete_thread(thread_id)


def get_chat_storage(client: AgentClient) -> ChatStorage:
    """Create a session-local adapter for an authenticated API client.

    Streamlit module globals are shared by browser sessions, so the adapter
    must never cache a user's session token globally.
    """
    return ChatStorage(client)
