"""Tests for sidebar chat-history helpers."""

from app.chat_storage import Chat
from app.sidebar import _format_chat_history_title


def test_format_chat_history_title_uses_thread_id_for_new_chat() -> None:
    chat = Chat(thread_id="087b99c9-1234-5678-90ab-cdef12345678", title="New Chat")

    display, full = _format_chat_history_title(chat)

    assert display == "Chat 087b99c9"
    assert full == "Chat 087b99c9"


def test_format_chat_history_title_collapses_whitespace_and_truncates() -> None:
    chat = Chat(
        thread_id="thread-1",
        title="  FAQ   Neutron   science knowledge base and troubleshooting guide  ",
    )

    display, full = _format_chat_history_title(chat, max_chars=22)

    assert full == "FAQ Neutron science knowledge base and troubleshooting guide"
    assert display.endswith("...")
    assert len(display) <= 22
