"""Tests for chat streaming display behavior."""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from juena.clients.client import AgentClientError
from juena.schema.server import ChatMessage
from juena.server.code_chat_inputs import CHAT_INPUT_MAX_CHARS, CHAT_INPUT_MAX_UPLOAD_MB


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


class _SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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


def test_build_agent_intro_message_lists_agents_and_repositories(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    client = Mock()
    client.list_repositories.return_value = [
        {"id": "datreat", "name": "DaTreat"},
        {"id": "jscatter", "name": "jscatter"},
    ]

    message = chat_interface.build_agent_intro_message(client)

    assert message.type == "ai"
    assert "`react_agent`" in message.content
    assert "`code_chat_agent`" in message.content
    assert "default starting agent" in message.content
    assert "Tavily" in message.content
    assert "Context7" in message.content
    assert "- `datreat`" in message.content
    assert "- `jscatter`" in message.content
    assert "Use the sidebar to switch agents at any time." in message.content


def test_build_agent_intro_message_falls_back_when_repositories_fail(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    client = Mock()
    client.list_repositories.side_effect = AgentClientError("boom")

    message = chat_interface.build_agent_intro_message(client)

    assert "Repository metadata is unavailable right now." in message.content


def test_render_chat_interface_inserts_intro_and_preserves_start_kickoff(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    render_header_mock = Mock()
    render_chat_input_styles_mock = Mock()
    render_message_mock = Mock()
    save_mock = Mock()
    placeholder = Mock()
    stream_calls: list[tuple[str, object, bool]] = []

    class StopRender(Exception):
        pass

    def fake_stream(message: str, message_placeholder, should_rerun: bool = True) -> None:
        stream_calls.append((message, message_placeholder, should_rerun))
        raise StopRender

    client = Mock()
    client.list_repositories.return_value = [{"id": "datreat", "name": "DaTreat"}]
    session_state = _SessionState(
        server_connected=True,
        client=client,
        messages=[],
        welcome_initialized=False,
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        show_system_messages=False,
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        chat_message=lambda _role: _DummyContext(),
        empty=lambda: placeholder,
        chat_input=Mock(return_value=None),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", render_header_mock)
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "save_message_to_storage", save_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", fake_stream)

    with pytest.raises(StopRender):
        chat_interface.render_chat_interface()

    assert session_state.welcome_initialized is True
    assert len(session_state.messages) == 1
    intro_message = session_state.messages[0]
    assert "`react_agent`" in intro_message.content
    assert "Tavily" in intro_message.content
    assert "- `datreat`" in intro_message.content
    render_header_mock.assert_called_once_with()
    render_chat_input_styles_mock.assert_called_once_with()
    render_message_mock.assert_called_once_with(intro_message, show_system=False)
    save_mock.assert_called_once_with("thread-1", intro_message)
    assert stream_calls == [("Start", placeholder, True)]


def test_render_chat_interface_does_not_duplicate_intro_for_existing_history(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    render_header_mock = Mock()
    render_chat_input_styles_mock = Mock()
    render_message_mock = Mock()
    save_mock = Mock()
    stream_mock = Mock()
    existing_message = ChatMessage(type="ai", content="Existing response")
    session_state = _SessionState(
        server_connected=True,
        client=Mock(),
        messages=[existing_message],
        welcome_initialized=False,
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        show_system_messages=False,
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(),
        chat_input=Mock(return_value=None),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", render_header_mock)
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "save_message_to_storage", save_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    render_header_mock.assert_called_once_with()
    render_chat_input_styles_mock.assert_called_once_with()
    render_message_mock.assert_called_once_with(existing_message, show_system=False)
    save_mock.assert_not_called()
    stream_mock.assert_not_called()
    assert session_state.messages == [existing_message]


def test_render_chat_interface_uses_attachment_enabled_chat_input_for_code_chat_agent(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    chat_input_mock = Mock(return_value=None)
    render_chat_input_styles_mock = Mock()
    session_state = _SessionState(
        server_connected=True,
        client=Mock(),
        messages=[ChatMessage(type="ai", content="Existing response")],
        welcome_initialized=True,
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        selected_agent="code_chat_agent",
        show_system_messages=False,
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(),
        chat_input=chat_input_mock,
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", Mock())
    monkeypatch.setattr(chat_interface, "save_message_to_storage", Mock())
    monkeypatch.setattr(chat_interface, "stream_and_display_response", Mock())

    chat_interface.render_chat_interface()

    chat_input_mock.assert_called_once_with(
        placeholder="Type your message here...",
        max_chars=CHAT_INPUT_MAX_CHARS,
        accept_file="multiple",
        max_upload_size=CHAT_INPUT_MAX_UPLOAD_MB,
        file_type=[
            "py", "js", "ts", "java", "c", "cpp", "rs", "go", "sh", "md", "txt",
            "log", "json", "yaml", "yml", "toml", "ini", "cfg", "csv", "sql",
            "html", "css", "xml", "ipynb",
        ],
    )
    render_chat_input_styles_mock.assert_called_once_with()


def test_render_chat_interface_streams_attachments_for_code_chat_agent(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    render_message_mock = Mock()
    render_chat_input_styles_mock = Mock()
    save_mock = Mock()
    stream_mock = Mock()
    attachment = SimpleNamespace(name="snippet.py")
    submission = SimpleNamespace(text="please inspect", files=[attachment])
    session_state = _SessionState(
        server_connected=True,
        client=Mock(),
        messages=[],
        welcome_initialized=True,
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        selected_agent="code_chat_agent",
        show_system_messages=False,
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(),
        chat_input=Mock(return_value=submission),
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "save_message_to_storage", save_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    assert session_state.messages[0].content == "please inspect\n\nAttachments:\n- snippet.py"
    render_message_mock.assert_called_once_with(session_state.messages[0])
    save_mock.assert_called_once_with("thread-1", session_state.messages[0])
    stream_mock.assert_called_once()
    render_chat_input_styles_mock.assert_called_once_with()
    assert stream_mock.call_args.args[:2] == ("please inspect", fake_st.empty.return_value)
    assert stream_mock.call_args.kwargs["attachments"] == [attachment]
