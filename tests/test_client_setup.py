"""Tests for Streamlit API client configuration."""

from app import client_setup


def test_initialize_client_uses_configured_timeout(monkeypatch) -> None:
    monkeypatch.setattr(client_setup.global_config, "TIMEOUT_SECONDS", 17)

    client = client_setup.initialize_client("http://localhost:8080")

    try:
        assert client.timeout == 17.0
    finally:
        client.close()
