"""Tests for AgentClient utility methods."""

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
