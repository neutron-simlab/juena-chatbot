"""Tests for chat streaming display behavior."""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

from juena.schema.server import ChatMessage


class _FakeTokenClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, **kwargs):
        yield from self._chunks

    def is_token_message(self, message):
        return isinstance(message, dict) and message.get("type") == "token"

    def get_token_content(self, message):
        if self.is_token_message(message):
            return message.get("content")
        return None


def _import_chat_interface(monkeypatch):
    fake_chat_storage = types.ModuleType("app.chat_storage")
    fake_chat_storage.get_chat_storage = Mock()

    monkeypatch.setitem(sys.modules, "app.chat_storage", fake_chat_storage)
    sys.modules.pop("app.chat_interface", None)
    return importlib.import_module("app.chat_interface")


def test_process_stream_chunk_ai_message_uses_finalize_renderer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
    save_mock = Mock()
    message_placeholder = Mock()
    messages: list[ChatMessage] = []
    chunk = ChatMessage(type="ai", content="final response")

    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(chat_interface, "save_message_to_storage", save_mock)

    response_text, received_complete = chat_interface.process_stream_chunk(
        chunk=chunk,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=message_placeholder,
        messages=messages,
        thread_id="thread-1",
    )

    assert response_text == "final response"
    assert received_complete is True
    assert messages == [chunk]
    finalize_mock.assert_called_once_with(message_placeholder, "final response")
    message_placeholder.markdown.assert_not_called()
    save_mock.assert_called_once_with("thread-1", chunk)


def test_stream_and_display_response_token_only_completion_uses_finalize_renderer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
    save_mock = Mock()
    render_streaming_token_mock = Mock()
    rerun_mock = Mock()
    message_placeholder = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient([{"type": "token", "content": "Hello"}]),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=rerun_mock,
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(chat_interface, "render_streaming_token", render_streaming_token_mock)
    monkeypatch.setattr(chat_interface, "save_message_to_storage", save_mock)

    chat_interface.stream_and_display_response(
        "hello",
        message_placeholder,
        should_rerun=False,
    )

    finalize_mock.assert_called_once_with(message_placeholder, "Hello")
    render_streaming_token_mock.assert_called_once_with("Hello", message_placeholder)
    message_placeholder.markdown.assert_not_called()
    rerun_mock.assert_not_called()
    assert len(session_state.messages) == 1
    assert session_state.messages[0].type == "ai"
    assert session_state.messages[0].content == "Hello"
    save_mock.assert_called_once_with("thread-1", session_state.messages[0])
