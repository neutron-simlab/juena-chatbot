"""Tests for sidebar chat-history helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import app.sidebar as sidebar
from app.chat_storage import Chat
from juena.clients.client import AgentAuthenticationError, AgentClientError


class _SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _make_session_state(**overrides) -> _SessionState:
    defaults = {
        "client": Mock(),
        "thread_id": "thread-1",
        "messages": ["current message"],
        "welcome_initialized": True,
    }
    defaults.update(overrides)
    return _SessionState(**defaults)


def test_format_chat_history_title_uses_thread_id_for_new_chat() -> None:
    chat = Chat(thread_id="087b99c9-1234-5678-90ab-cdef12345678", title="New Chat")

    display, full = sidebar._format_chat_history_title(chat)

    assert display == "Chat 087b99c9"
    assert full == "Chat 087b99c9"


def test_format_chat_history_title_collapses_whitespace_and_truncates() -> None:
    chat = Chat(
        thread_id="thread-1",
        title="  FAQ   Neutron   science knowledge base and troubleshooting guide  ",
    )

    display, full = sidebar._format_chat_history_title(chat, max_chars=22)

    assert full == "FAQ Neutron science knowledge base and troubleshooting guide"
    assert display.endswith("...")
    assert len(display) <= 22


def test_logout_form_posts_directly_to_public_auth_endpoint() -> None:
    html = sidebar._logout_form_html("https://juena.example.org/")

    assert 'action="https://juena.example.org/auth/logout"' in html
    assert 'method="post"' in html
    assert "Sign out" in html
    assert "javascript" not in html.lower()


def test_logout_form_escapes_configured_public_url() -> None:
    html = sidebar._logout_form_html('https://juena.example.org/" onsubmit="bad')

    assert 'onsubmit="bad' not in html
    assert "&quot;" in html


def test_delete_current_chat_selects_most_recent_remaining_chat(monkeypatch) -> None:
    remaining_chat = Chat(thread_id="thread-2", title="Existing Chat")
    remaining_messages = ["loaded message"]
    storage = Mock()
    storage.list_chats.return_value = [remaining_chat]
    storage.load_messages.return_value = remaining_messages
    session_state = _make_session_state(welcome_initialized=False)
    fake_st = SimpleNamespace(session_state=session_state, query_params={})

    monkeypatch.setattr(sidebar, "st", fake_st)

    deleted, error_message = sidebar._delete_chat_with_server_state(
        storage,
        "thread-1",
        is_current=True,
    )

    assert deleted is True
    assert error_message is None
    session_state.client.delete_thread.assert_not_called()
    storage.delete_chat.assert_called_once_with("thread-1")
    storage.list_chats.assert_called_once_with(limit=1)
    storage.load_messages.assert_called_once_with("thread-2")
    storage.upsert_chat.assert_not_called()
    assert session_state.thread_id == "thread-2"
    assert session_state.messages == remaining_messages
    assert session_state.welcome_initialized is True


def test_delete_last_current_chat_creates_single_replacement_chat(monkeypatch) -> None:
    storage = Mock()
    storage.list_chats.return_value = []
    session_state = _make_session_state()
    fake_st = SimpleNamespace(session_state=session_state, query_params={})

    monkeypatch.setattr(sidebar, "st", fake_st)
    monkeypatch.setattr(sidebar, "uuid4", lambda: "replacement-thread")

    deleted, error_message = sidebar._delete_chat_with_server_state(
        storage,
        "thread-1",
        is_current=True,
    )

    assert deleted is True
    assert error_message is None
    session_state.client.delete_thread.assert_not_called()
    storage.delete_chat.assert_called_once_with("thread-1")
    storage.list_chats.assert_called_once_with(limit=1)
    storage.load_messages.assert_not_called()
    # The blank replacement is not persisted; the server creates the row when
    # the first message is sent, so unused chats never reach the history list.
    storage.upsert_chat.assert_not_called()
    assert session_state.thread_id == "replacement-thread"
    assert fake_st.query_params["thread"] == "replacement-thread"
    assert session_state.messages == []
    assert session_state.welcome_initialized is False


def test_delete_non_current_chat_leaves_active_session_unchanged(monkeypatch) -> None:
    storage = Mock()
    session_state = _make_session_state()
    fake_st = SimpleNamespace(session_state=session_state, query_params={})

    monkeypatch.setattr(sidebar, "st", fake_st)

    deleted, error_message = sidebar._delete_chat_with_server_state(
        storage,
        "thread-2",
        is_current=False,
    )

    assert deleted is True
    assert error_message is None
    session_state.client.delete_thread.assert_not_called()
    storage.delete_chat.assert_called_once_with("thread-2")
    storage.list_chats.assert_not_called()
    storage.load_messages.assert_not_called()
    storage.upsert_chat.assert_not_called()
    assert session_state.thread_id == "thread-1"
    assert session_state.messages == ["current message"]
    assert session_state.welcome_initialized is True


def test_research_status_reruns_app_when_session_expired(monkeypatch) -> None:
    client = Mock()
    client.list_research.side_effect = AgentAuthenticationError("expired")
    fake_st = SimpleNamespace(
        session_state=_make_session_state(client=client),
        rerun=Mock(),
        caption=Mock(),
    )
    monkeypatch.setattr(sidebar, "st", fake_st)

    sidebar._render_research_status_content()

    client.list_research.assert_called_once_with("thread-1")
    fake_st.rerun.assert_called_once_with(scope="app")
    fake_st.caption.assert_not_called()


def test_research_status_reports_polling_failure(monkeypatch) -> None:
    client = Mock()
    client.list_research.side_effect = AgentClientError("server unavailable")
    fake_st = SimpleNamespace(
        session_state=_make_session_state(client=client),
        rerun=Mock(),
        caption=Mock(),
    )
    monkeypatch.setattr(sidebar, "st", fake_st)

    sidebar._render_research_status_content()

    fake_st.rerun.assert_not_called()
    fake_st.caption.assert_called_once_with(
        "⚠️ Background research status unavailable"
    )


def test_research_status_renders_running_and_ready_badges(monkeypatch) -> None:
    client = Mock()
    client.list_research.return_value = {
        "jobs": [],
        "running": 1,
        "ready": 2,
    }
    fake_st = SimpleNamespace(
        session_state=_make_session_state(client=client),
        subheader=Mock(),
        caption=Mock(),
        button=Mock(return_value=False),
        divider=Mock(),
    )
    monkeypatch.setattr(sidebar, "st", fake_st)

    sidebar._render_research_status_content()

    fake_st.subheader.assert_called_once_with("Background research")
    fake_st.caption.assert_called_once_with("⏳ 1 running")
    fake_st.button.assert_called_once_with(
        "📥 2 results ready",
        width="stretch",
        key="collect-research",
    )
    fake_st.divider.assert_called_once_with()
