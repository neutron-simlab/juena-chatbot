"""Tests for multipart file-backed API endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from juena.server import api_endpoints


class _FakeAgent:
    def __init__(self) -> None:
        self.last_kwargs = None

    async def aget_state(self, config):  # noqa: ANN001
        return SimpleNamespace(values={"files": {"/inputs/old.txt": {"content": ["old"]}}})

    async def ainvoke(self, **kwargs):
        self.last_kwargs = kwargs
        return [("values", {"messages": [AIMessage(content="ready")]})]


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(api_endpoints.router)
    return TestClient(app)


def test_invoke_with_files_stages_inputs_and_returns_chat_message(monkeypatch) -> None:
    fake_agent = _FakeAgent()

    async def fake_get_agent(agent_id: str, provider: str | None = None, model: str | None = None):
        assert agent_id == "code_chat_agent"
        assert provider == "openai"
        assert model == "gpt-test"
        return fake_agent

    monkeypatch.setattr(api_endpoints, "get_agent", fake_get_agent)

    client = _build_test_client()
    response = client.post(
        "/code_chat_agent/invoke_with_files",
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

    client = _build_test_client()
    response = client.post(
        "/code_chat_agent/invoke_with_files",
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

    client = _build_test_client()
    response = client.delete("/threads/thread-123")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "thread_id": "thread-123",
        "message": "Thread thread-123 deleted successfully",
    }
    assert deleted == {"thread_id": "thread-123"}
