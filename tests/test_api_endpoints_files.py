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


def test_invoke_with_files_stages_inputs_and_returns_chat_message(monkeypatch) -> None:
    fake_agent = _FakeAgent()

    async def fake_get_agent(agent_id: str, provider: str | None = None, model: str | None = None):
        assert agent_id == "juena"
        assert provider == "openai"
        assert model == "gpt-test"
        return fake_agent

    monkeypatch.setattr(api_endpoints, "get_agent", fake_get_agent)

    client = _build_test_client(monkeypatch)
    response = client.post(
        "/juena/invoke_with_files",
        data={
            "message": "Please inspect this upload",
            "thread_id": "thread-1",
            "provider": "openai",
            "model": "gpt-test",
        },
        files=[
            ("attachments", ("snippet.py", b"print('hello')\n", "text/x-python")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "ai"
    assert payload["content"] == "ready"
    assert payload["thread_id"] == "thread-1"

    assert fake_agent.last_kwargs is not None
    assert "Persistent uploaded files for this chat are available under `/inputs/uploads/`." in (
        fake_agent.last_kwargs["input"]["messages"][0].content
    )
    assert fake_agent.last_kwargs["input"]["files"]["/inputs/old.txt"] is None
    assert "/inputs/current_message.txt" in fake_agent.last_kwargs["input"]["files"]
    assert "/inputs/uploads/snippet.py" in fake_agent.last_kwargs["input"]["files"]
    assert "/inputs/uploads_manifest.md" in fake_agent.last_kwargs["input"]["files"]


def test_invoke_with_files_rejects_unsupported_file_types(monkeypatch) -> None:
    async def fake_get_agent(agent_id: str, provider: str | None = None, model: str | None = None):
        return _FakeAgent()

    monkeypatch.setattr(api_endpoints, "get_agent", fake_get_agent)

    client = _build_test_client(monkeypatch)
    response = client.post(
        "/juena/invoke_with_files",
        data={"message": "Please inspect this upload"},
        files=[
            ("attachments", ("report.pdf", b"%PDF-1.7", "application/pdf")),
        ],
    )

    assert response.status_code == 400
    assert "unsupported extension" in response.json()["detail"]


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


def test_agents_endpoint_reports_the_registry_default(monkeypatch) -> None:
    """`GET /agents` must read the default at request time.

    The default is rebound when an agent self-registers, which happens after
    this module is imported. A `from ... import DEFAULT_AGENT` in the endpoint
    module would freeze whatever value was current at import.
    """

    monkeypatch.setattr(api_endpoints, "list_registered_agents", lambda: ["late-agent"])
    monkeypatch.setattr(api_endpoints, "get_default_agent", lambda: "late-agent")

    client = _build_test_client(monkeypatch)
    response = client.get("/agents")

    assert response.status_code == 200
    assert response.json() == {"agents": ["late-agent"], "default": "late-agent"}


def test_invoke_without_an_agent_id_resolves_the_current_default(monkeypatch) -> None:
    """`POST /invoke` has no `{agent_id}` path segment, so it must fall back."""

    seen: dict[str, str] = {}

    async def fake_get_agent(agent_id: str, provider: str | None = None, model: str | None = None):
        seen["agent_id"] = agent_id
        return _FakeAgent()

    monkeypatch.setattr(api_endpoints, "get_agent", fake_get_agent)
    monkeypatch.setattr(api_endpoints, "get_default_agent", lambda: "late-agent")

    client = _build_test_client(monkeypatch)
    response = client.post("/invoke", json={"message": "hello", "thread_id": "thread-1"})

    assert response.status_code == 200
    assert seen["agent_id"] == "late-agent"
