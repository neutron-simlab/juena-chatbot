"""Tests for the authenticated Streamlit chat-storage adapter."""

from app.chat_storage import get_chat_storage
from juena.clients.client import AgentClient


def test_chat_storage_never_shares_clients_between_browser_sessions() -> None:
    alice = AgentClient(session_token="alice-session")
    bob = AgentClient(session_token="bob-session")

    try:
        alice_storage = get_chat_storage(alice)
        bob_storage = get_chat_storage(bob)

        assert alice_storage is not bob_storage
        assert alice_storage.client.session_token == "alice-session"
        assert bob_storage.client.session_token == "bob-session"
    finally:
        alice.close()
        bob.close()
