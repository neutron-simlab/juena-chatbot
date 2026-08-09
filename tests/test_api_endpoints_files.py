"""Tests for multipart file-backed API endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from juena.server.api import endpoints as api_endpoints
from juena.server.auth import AuthenticatedUser, get_current_user
from juena.server.database import get_db_session


class _FakeAgent:
    def __init__(self) -> None:
        self.last_kwargs = None

    async def aget_state(self, config):  # noqa: ANN001
        return SimpleNamespace(values={"files": {"/inputs/old.txt": {"content": ["old"]}}})

    async def ainvoke(self, **kwargs):
        self.last_kwargs = kwargs
        return [("values", {"messages": [AIMessage(content="ready")]})]


class _FakeSession:
    async def commit(self) -> None:
        return None

    async def delete(self, value) -> None:  # noqa: ANN001
        return None


@pytest.mark.asyncio
async def test_authorizing_a_message_touches_chat_recency(monkeypatch) -> None:
    old_updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    chat = SimpleNamespace(thread_id="thread-1", updated_at=old_updated_at)
    committed = False

    async def fake_ensure_owned_chat(session, owner_id, thread_id):  # noqa: ANN001
        return chat

    class Session:
        async def commit(self) -> None:
            nonlocal committed
            committed = True

    monkeypatch.setattr(api_endpoints, "ensure_owned_chat", fake_ensure_owned_chat)
    user = AuthenticatedUser(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        saml_subject="subject-1",
        email=None,
        display_name=None,
    )

    thread_id = await api_endpoints._authorize_thread(Session(), user, "thread-1")

    assert thread_id == "thread-1"
    assert chat.updated_at > old_updated_at
    assert committed is True


def _build_test_client(monkeypatch) -> TestClient:  # noqa: ANN001
    app = FastAPI()
    app.include_router(api_endpoints.router)
    user_id = UUID("00000000-0000-0000-0000-000000000001")

    async def fake_user() -> AuthenticatedUser:
        return AuthenticatedUser(
            id=user_id,
            saml_subject="subject-1",
            email=None,
            display_name=None,
        )

    async def fake_session():
        yield _FakeSession()

    async def fake_ensure_owned_chat(session, owner_id, thread_id):  # noqa: ANN001
        return SimpleNamespace(
            thread_id=thread_id or "generated-thread",
            user_id=owner_id,
        )

    async def fake_get_owned_chat(session, owner_id, thread_id):  # noqa: ANN001
        return SimpleNamespace(thread_id=thread_id, user_id=owner_id)

    app.dependency_overrides[get_current_user] = fake_user
    app.dependency_overrides[get_db_session] = fake_session
    monkeypatch.setattr(api_endpoints, "ensure_owned_chat", fake_ensure_owned_chat)
    monkeypatch.setattr(api_endpoints, "get_owned_chat", fake_get_owned_chat)
    return TestClient(app)


def test_delete_thread_endpoint_removes_persisted_state(monkeypatch) -> None:
    deleted: dict[str, str] = {}

    class _FakeCheckpointer:
        async def adelete_thread(self, thread_id: str) -> None:
            deleted["thread_id"] = thread_id

    monkeypatch.setattr(api_endpoints, "get_checkpointer", lambda: _FakeCheckpointer())

    client = _build_test_client(monkeypatch)
    response = client.delete("/threads/thread-123")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "thread_id": "thread-123",
        "message": "Thread thread-123 deleted successfully",
    }
    assert deleted == {"thread_id": "thread-123"}
