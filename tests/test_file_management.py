"""Tests for Streamlit API client configuration."""

from app import file_management


def test_initialize_client_uses_configured_timeout(monkeypatch) -> None:
    monkeypatch.setattr(file_management.global_config, "TIMEOUT_SECONDS", 17)

    client = file_management.initialize_client("http://localhost:8080")

    try:
        assert client.timeout == 17.0
    finally:
        client.close()
