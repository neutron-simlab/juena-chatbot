"""Tests for AgentClient utility methods."""

from types import SimpleNamespace

import httpx
import pytest

from juena.clients import client as client_module
from juena_core.schema.server import ChatMessage


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

    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena", timeout=5.0)
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
        agent="juena",
        session_token="opaque-session",
    )

    assert client._headers == {"Cookie": "juena_session=opaque-session"}


def test_list_research_sends_session_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return _FakeResponse({"jobs": [], "running": 1, "ready": 0})

    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        session_token="opaque-session",
        timeout=5.0,
    )
    monkeypatch.setattr(client._client, "get", fake_get)

    status = client.list_research("thread-1")

    assert status == {"jobs": [], "running": 1, "ready": 0}
    assert captured == {
        "url": "http://localhost:8080/research",
        "kwargs": {
            "params": {"thread_id": "thread-1"},
            "headers": {"Cookie": "juena_session=opaque-session"},
            "timeout": 5.0,
        },
    }


def test_list_research_does_not_disguise_unauthorized_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "http://localhost:8080/research")
    response = httpx.Response(
        401,
        request=request,
        json={"detail": "Invalid or expired session"},
    )
    client = client_module.AgentClient(base_url="http://localhost:8080")
    monkeypatch.setattr(client._client, "get", lambda *_args, **_kwargs: response)

    with pytest.raises(
        client_module.AgentAuthenticationError,
        match="Invalid or expired session",
    ):
        client.list_research("thread-1")


def test_close_closes_reusable_http_client() -> None:
    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena")

    client.close()

    assert client._client.is_closed


def test_sse_parser_preserves_approval_choices_and_artifact_bytes() -> None:
    client = client_module.AgentClient(base_url="http://localhost:8080", agent="juena")

    approval = client._parse_sse_data(
        '{"type":"approval_required","interrupt_id":"i-1","command":"python x.py",'
        '"allowed_decisions":["approve","reject"],"limits":{"network":"none"}}'
    )
    artifact = client._parse_sse_data(
        '{"type":"artifact","artifact_id":"a-1","filename":"plot.png",'
        '"mime_type":"image/png","kind":"image","content_base64":"abc"}'
    )

    assert approval["allowed_decisions"] == ["approve", "reject"]
    assert approval["command"] == "python x.py"
    assert artifact["artifact_id"] == "a-1"
    assert artifact["content_base64"] == "abc"


def test_resume_stream_posts_exact_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_stream(method: str, url: str, **kwargs):
        captured.update(method=method, url=url, kwargs=kwargs)
        return _FakeStreamContext(
            _FakeStreamResponse(
                [
                    'data: {"type":"token","content":"done"}',
                ]
            )
        )

    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="juena",
        timeout=5.0,
        session_token="session",
    )
    monkeypatch.setattr(client._client, "stream", fake_stream)

    chunks = list(
        client.resume_stream(
            thread_id="thread-1",
            interrupt_id="interrupt-1",
            decision="edit",
            edited_command="python revised.py",
            provider="openai",
            model="gpt-4o-mini",
        )
    )

    assert chunks == [{"type": "token", "content": "done"}]
    assert captured["url"] == "http://localhost:8080/juena/resume"
    assert captured["kwargs"]["json"] == {
        "kind": "execute_approval",
        "thread_id": "thread-1",
        "interrupt_id": "interrupt-1",
        "decision": "edit",
        "edited_command": "python revised.py",
        "provider": "openai",
        "model": "gpt-4o-mini",
    }


def test_resume_stream_posts_an_answer_without_a_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_stream(method: str, url: str, **kwargs):
        captured.update(method=method, url=url, kwargs=kwargs)
        return _FakeStreamContext(
            _FakeStreamResponse(['data: {"type":"token","content":"done"}'])
        )

    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="juena",
        timeout=5.0,
        session_token="session",
    )
    monkeypatch.setattr(client._client, "stream", fake_stream)

    list(
        client.resume_stream(
            thread_id="thread-1",
            interrupt_id="interrupt-2",
            answer="Empty cell",
        )
    )

    assert captured["kwargs"]["json"] == {
        "kind": "clarification",
        "thread_id": "thread-1",
        "interrupt_id": "interrupt-2",
        "answer": "Empty cell",
    }


def test_resume_stream_rejects_an_ambiguous_reply() -> None:
    client = client_module.AgentClient(
        base_url="http://localhost:8080",
        agent="juena",
        timeout=5.0,
        session_token="session",
    )

    with pytest.raises(client_module.AgentClientError, match="exactly one"):
        list(client.resume_stream(thread_id="t", interrupt_id="i"))
