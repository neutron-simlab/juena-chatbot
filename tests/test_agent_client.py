"""Tests for the two routes JüNA's ``AgentClient`` adds to core's client.

Everything behind a route juena-core serves -- the streaming transport, the SSE
parser, chats, artifacts, interrupts and thread cleanup -- moved to core's
``test_client_contracts.py`` in plan 02/step 3, because the subclass does not
touch any of it. What is left here is what decision 10 says belongs to this
application: ``/auth/me`` and ``/research`` are JüNA's routes, so their client
methods are JüNA's too.
"""

import httpx
import pytest

from juena.clients import client as client_module


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


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
