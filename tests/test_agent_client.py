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
    """Minimal stand-in for an SSE response, as httpx_sse expects to see it."""

    def __init__(self, lines):
        self._lines = lines
        self.headers = {"content-type": "text/event-stream"}

    def raise_for_status(self) -> None:
        return None

    def iter_text(self):
        # Each SSE event is terminated by a blank line.
        for line in self._lines:
            yield f"{line}\n\n"


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

    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="react_agent",
        timeout=5.0,
    )
    monkeypatch.setattr(client._client, "get", fake_get)

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

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent")
    monkeypatch.setattr(client._client, "get", fake_get)

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

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena", timeout=5.0)
    monkeypatch.setattr(client._client, "post", fake_post)
    attachment = SimpleNamespace(
        name="snippet.py",
        type="text/x-python",
        getvalue=lambda: b"print('hello')\n",
    )

    message = client.invoke(
        "debug this",
        thread_id="thread-1",
        attachments=[attachment],
    )

    assert isinstance(message, ChatMessage)
    assert captured["url"] == "http://localhost:8080/juena/invoke_with_files"
    assert captured["data"] == {
        "message": "debug this",
        "thread_id": "thread-1",
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

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent", timeout=5.0)
    monkeypatch.setattr(client._client, "post", fake_post)
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
                ]
            )
        )

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena", timeout=5.0)
    monkeypatch.setattr(client._client, "stream", fake_stream)
    attachment = SimpleNamespace(
        name="trace.log",
        type="text/plain",
        getvalue=lambda: b"ERROR: boom\n",
    )

    chunks = list(client.stream("please inspect", attachments=[attachment]))

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChatMessage)
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8080/juena/stream_with_files"
    assert captured["kwargs"]["data"] == {"message": "please inspect"}
    assert captured["kwargs"]["files"] == [
        ("attachments", ("trace.log", b"ERROR: boom\n", "text/plain"))
    ]
    stream_timeout = captured["kwargs"]["timeout"]
    assert isinstance(stream_timeout, httpx.Timeout)
    assert stream_timeout.connect == 5.0
    assert stream_timeout.read is None
    assert stream_timeout.write == 5.0
    assert stream_timeout.pool == 5.0


def test_delete_thread_uses_thread_cleanup_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_delete(url: str, *, headers: dict[str, str], timeout: float | None):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse({"status": "success", "thread_id": "thread-1"})

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="react_agent", timeout=5.0)
    monkeypatch.setattr(client._client, "delete", fake_delete)
    payload = client.delete_thread("thread-1")

    assert payload == {"status": "success", "thread_id": "thread-1"}
    assert captured == {
        "url": "http://localhost:8080/threads/thread-1",
        "headers": {},
        "timeout": 5.0,
    }


def test_session_cookie_replaces_shared_auth_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_SECRET", "must-not-be-used")
    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="react_agent",
        session_token="opaque-session",
    )

    assert client._headers == {"Cookie": "juena_session=opaque-session"}


def test_close_closes_reusable_http_client() -> None:
    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena")

    client.close()

    assert client._client.is_closed
