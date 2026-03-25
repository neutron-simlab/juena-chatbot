"""Tests for AgentClient utility methods."""

from types import SimpleNamespace

import httpx
import pytest

from juena.clients import client as client_module
from juena.schema.server import ChatMessage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, exc_type, exc, tb):
        return False


def test_list_repositories_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTH_SECRET", raising=False)
    captured: dict[str, object] = {}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float | None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse([{"id": "datreat", "name": "DaTreat"}])

    monkeypatch.setattr(client_module.httpx, "get", fake_get)

    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="react_agent",
        timeout=5.0,
    )

    repositories = client.list_repositories()

    assert repositories == [{"id": "datreat", "name": "DaTreat"}]
    assert captured == {
        "url": "http://localhost:8080/repositories",
        "headers": {},
        "timeout": 5.0,
    }


def test_list_repositories_raises_client_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTH_SECRET", raising=False)

    def fake_get(url: str, *, headers: dict[str, str], timeout: float | None):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(client_module.httpx, "get", fake_get)

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent")

    with pytest.raises(client_module.AgentClientError, match="boom"):
        client.list_repositories()


def test_invoke_uses_multipart_endpoint_when_attachments_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, data, files, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse({"type": "ai", "content": "ok"})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="code_chat_agent", timeout=5.0)
    attachment = SimpleNamespace(
        name="snippet.py",
        type="text/x-python",
        getvalue=lambda: b"print('hello')\n",
    )

    message = client.invoke(
        "debug this",
        thread_id="thread-1",
        user_id="user-1",
        attachments=[attachment],
    )

    assert isinstance(message, ChatMessage)
    assert captured["url"] == "http://localhost:8080/code_chat_agent/invoke_with_files"
    assert captured["data"] == {
        "message": "debug this",
        "thread_id": "thread-1",
        "user_id": "user-1",
    }
    assert captured["files"] == [
        ("attachments", ("snippet.py", b"print('hello')\n", "text/x-python"))
    ]


def test_invoke_without_attachments_uses_json_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse({"type": "ai", "content": "ok"})

    monkeypatch.setattr(client_module.httpx, "post", fake_post)

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent", timeout=5.0)
    client.invoke("hello", thread_id="thread-1")

    assert captured["url"] == "http://localhost:8080/react_agent/invoke"
    assert captured["json"]["message"] == "hello"
    assert captured["json"]["thread_id"] == "thread-1"


def test_stream_uses_multipart_endpoint_when_attachments_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_stream(method: str, url: str, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeStreamContext(
            _FakeStreamResponse(
                [
                    'data: {"type":"message","content":{"type":"ai","content":"ok","tool_calls":[],"tool_call_id":null,"run_id":null,"thread_id":null,"response_metadata":{},"custom_data":{},"timestamp":null}}',
                    "data: [DONE]",
                ]
            )
        )

    monkeypatch.setattr(client_module.httpx, "stream", fake_stream)

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="code_chat_agent", timeout=5.0)
    attachment = SimpleNamespace(
        name="trace.log",
        type="text/plain",
        getvalue=lambda: b"ERROR: boom\n",
    )

    chunks = list(client.stream("please inspect", attachments=[attachment]))

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatMessage)
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8080/code_chat_agent/stream_with_files"
    assert captured["kwargs"]["data"] == {"message": "please inspect"}
    assert captured["kwargs"]["files"] == [
        ("attachments", ("trace.log", b"ERROR: boom\n", "text/plain"))
    ]


def test_delete_thread_uses_thread_cleanup_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_delete(url: str, *, headers: dict[str, str], timeout: float | None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse({"status": "success", "thread_id": "thread-1"})

    monkeypatch.setattr(client_module.httpx, "delete", fake_delete)

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent", timeout=5.0)
    payload = client.delete_thread("thread-1")

    assert payload == {"status": "success", "thread_id": "thread-1"}
    assert captured == {
        "url": "http://localhost:8080/threads/thread-1",
        "headers": {},
        "timeout": 5.0,
    }
