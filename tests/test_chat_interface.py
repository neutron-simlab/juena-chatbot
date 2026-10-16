"""What the JüNA chat page does around core's stream loop.

The stream loop itself, the chunk classifiers, the status container and the two
interrupt cards moved to juena-core in plan 02/step 3 -- they implement the SSE
contract core's server emits, so core is where they have to agree with it
(00-BOUNDARY.md, decision 7). juena-core's ``test_ui_streaming.py`` holds them.

What is tested here is the page: which starter chips it offers, what it refuses
to upload, and the fact that it draws this product's header before any of it.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from juena_core.schema.server import ChatMessage
from juena_core.schema.upload_limits import CHAT_INPUT_MAX_CHARS, CHAT_INPUT_MAX_UPLOAD_MB
from app.starters import StarterPrompt as chat_interface_starter


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
    fake_chat_storage = types.ModuleType("juena_core.ui.chat_storage")
    fake_chat_storage.get_chat_storage = Mock()

    monkeypatch.setitem(sys.modules, "juena_core.ui.chat_storage", fake_chat_storage)
    sys.modules.pop("app.chat_interface", None)
    return importlib.import_module("app.chat_interface")


def test_render_starter_topics_returns_selected_prompt(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    starters = [
        chat_interface_starter("Datreat", "What is the DaTreat repository for?"),
        chat_interface_starter("Jscatter", "What is the jscatter repository for?"),
    ]
    monkeypatch.setattr(chat_interface, "build_starter_prompts", lambda: starters)
    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(
            caption=Mock(),
            pills=Mock(return_value="Jscatter"),
            session_state=_SessionState(thread_id="thread-1"),
        ),
    )

    assert chat_interface.render_starter_topics() == (
        "What is the jscatter repository for?"
    )


def test_render_starter_topics_returns_none_without_selection(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface,
        "build_starter_prompts",
        lambda: [chat_interface_starter("Datreat", "prompt")],
    )
    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(
            caption=Mock(),
            pills=Mock(return_value=None),
            session_state=_SessionState(thread_id="thread-1"),
        ),
    )

    assert chat_interface.render_starter_topics() is None


def test_plot_starter_preview_does_not_send_when_sandbox_is_disabled(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    preview = chat_interface_starter(
        "Plot with Python",
        "plot prompt",
        requires_sandbox=True,
    )
    info_mock = Mock()
    monkeypatch.setattr(chat_interface.global_config, "SANDBOX_ENABLED", False)
    monkeypatch.setattr(chat_interface, "build_starter_prompts", lambda: [preview])
    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(
            caption=Mock(),
            pills=Mock(return_value="Plot with Python"),
            info=info_mock,
            session_state=_SessionState(thread_id="thread-1"),
        ),
    )

    assert chat_interface.render_starter_topics() is None
    info_mock.assert_called_once()


def test_render_chat_interface_offers_starter_topics_on_empty_chat(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    render_header_mock = Mock()
    render_chat_input_styles_mock = Mock()
    render_message_mock = Mock()
    stream_mock = Mock()
    starter_mock = Mock(return_value=None)

    session_state = _SessionState(
        client=Mock(),
        messages=[],
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
        container=lambda: _DummyContext(),
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", render_header_mock)
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "render_starter_topics", starter_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    starter_mock.assert_called_once_with()
    # Nothing picked and nothing typed: no message, no stream, no "Start" kickoff.
    assert session_state.messages == []
    stream_mock.assert_not_called()
    render_header_mock.assert_called_once_with()
    render_chat_input_styles_mock.assert_called_once_with()


def test_render_chat_interface_sends_selected_starter_topic(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    render_message_mock = Mock()
    stream_mock = Mock()
    session_state = _SessionState(
        client=Mock(),
        messages=[],
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
        container=lambda: _DummyContext(),
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", Mock())
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(
        chat_interface,
        "render_starter_topics",
        Mock(return_value="What is the jscatter repository for?"),
    )
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    # The chip's question is shown verbatim as the user's own message.
    assert len(session_state.messages) == 1
    assert session_state.messages[0].type == "human"
    assert session_state.messages[0].content == "What is the jscatter repository for?"
    assert stream_mock.call_args.args[0] == "What is the jscatter repository for?"


def test_render_chat_interface_does_not_duplicate_intro_for_existing_history(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    render_header_mock = Mock()
    render_chat_input_styles_mock = Mock()
    render_message_mock = Mock()
    stream_mock = Mock()
    existing_message = ChatMessage(type="ai", content="Existing response")
    session_state = _SessionState(
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
        container=lambda: _DummyContext(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", render_header_mock)
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    render_header_mock.assert_called_once_with()
    render_chat_input_styles_mock.assert_called_once_with()
    render_message_mock.assert_called_once_with(existing_message, show_system=False)
    stream_mock.assert_not_called()
    assert session_state.messages == [existing_message]


@pytest.mark.parametrize("mention_thread", [None, "thread-1", "other-thread"])
def test_render_chat_interface_uses_attachment_enabled_chat_input(monkeypatch, mention_thread) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    chat_input_mock = Mock(return_value=None)
    render_chat_input_styles_mock = Mock()
    session_state = _SessionState(
        client=Mock(),
        messages=[ChatMessage(type="ai", content="Existing response")],
        welcome_initialized=True,
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
        chat_input=chat_input_mock,
        container=lambda: _DummyContext(),
    )
    if mention_thread:
        session_state.filesystem_mention = {"thread_id": mention_thread, "path": "/workspace/analysis.py"}

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", Mock())
    monkeypatch.setattr(chat_interface, "stream_and_display_response", Mock())

    chat_interface.render_chat_interface()

    if mention_thread == "thread-1":
        assert session_state.chat_prompt == "Please look at `/workspace/analysis.py`: "
    else:
        assert "chat_prompt" not in session_state
    assert "filesystem_mention" not in session_state
    chat_input_mock.assert_called_once_with(
        placeholder="Type your message here...",
        key="chat_prompt",
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


def test_render_chat_interface_streams_attachments(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    render_message_mock = Mock()
    render_chat_input_styles_mock = Mock()
    stream_mock = Mock()
    attachment = SimpleNamespace(name="snippet.py", size=128)
    submission = SimpleNamespace(text="please inspect", files=[attachment])
    session_state = _SessionState(
        client=Mock(),
        messages=[],
        welcome_initialized=True,
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
        chat_input=Mock(return_value=submission),
        container=lambda: _DummyContext(),
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "render_starter_topics", Mock(return_value=None))
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    # The typed text stays the message; attachments ride in custom_data so the
    # same rendering path works for a reloaded thread.
    assert session_state.messages[0].content == "please inspect"
    assert session_state.messages[0].custom_data["attachments"] == [{"name": "snippet.py"}]
    render_message_mock.assert_called_once_with(session_state.messages[0])
    stream_mock.assert_called_once()
    render_chat_input_styles_mock.assert_called_once_with()
    assert stream_mock.call_args.args[:2] == ("please inspect", fake_st.empty.return_value)
    assert stream_mock.call_args.kwargs["attachments"] == [attachment]


def test_a_new_turn_is_drawn_above_the_composer(monkeypatch) -> None:
    """In the chat column the composer is drawn where it is called, not pinned
    to the bottom of the window. The turn it submits arrives after that call,
    so it has to go into a block made before the composer, or the box sits
    above the message and the live reply until the stream ends."""
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    events: list[tuple] = []
    open_blocks: list[int] = []
    created: list[int] = []

    class _Block:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self):
            open_blocks.append(self.index)
            return self

        def __exit__(self, exc_type, exc, tb):
            open_blocks.pop()
            return False

    def container():
        created.append(len(created))
        events.append(("block", created[-1]))
        return _Block(created[-1])

    def chat_input(**_kwargs):
        events.append(("composer",))
        return "fit the SANS data"

    def chat_message(role):
        events.append(("bubble", role, tuple(open_blocks)))
        return _DummyContext()

    def render_message(message, **_kwargs):
        events.append(("message", message.content, tuple(open_blocks)))

    existing_message = ChatMessage(type="ai", content="Existing response")
    session_state = _SessionState(
        client=Mock(),
        messages=[existing_message],
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        show_system_messages=False,
    )
    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(
            session_state=session_state,
            chat_message=chat_message,
            empty=Mock(),
            chat_input=chat_input,
            container=container,
            error=Mock(),
        ),
    )
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", Mock())
    monkeypatch.setattr(chat_interface, "render_message", render_message)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", Mock())

    chat_interface.render_chat_interface()

    assert events.index(("block", 0)) < events.index(("composer",))
    assert ("message", "Existing response", (0,)) in events
    assert ("message", "fit the SANS data", (0,)) in events
    assert ("bubble", "assistant", (0,)) in events


def test_render_chat_interface_preserves_raw_math_in_user_message(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    render_message_mock = Mock()
    stream_mock = Mock()
    submission = r"Show \(a^2+b^2=c^2\)"
    session_state = _SessionState(
        client=Mock(),
        messages=[],
        welcome_initialized=True,
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
        chat_input=Mock(return_value=submission),
        container=lambda: _DummyContext(),
        error=Mock(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", Mock())
    monkeypatch.setattr(chat_interface, "render_message", render_message_mock)
    monkeypatch.setattr(chat_interface, "render_starter_topics", Mock(return_value=None))
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    assert len(session_state.messages) == 1
    assert session_state.messages[0].content == submission
    render_message_mock.assert_called_once_with(session_state.messages[0])
    assert stream_mock.call_args.args[:2] == (submission, fake_st.empty.return_value)


@pytest.mark.parametrize(
    ("files", "expected_error"),
    [
        (
            [SimpleNamespace(name=f"f{index}.py", size=10) for index in range(6)],
            "At most 5 files",
        ),
        ([SimpleNamespace(name="huge.py", size=2 * 1024 * 1024)], "over the 1 MB"),
        ([SimpleNamespace(name="archive.zip", size=10)], "unsupported extension"),
    ],
)
def test_render_chat_interface_refuses_bad_uploads_without_touching_history(
    monkeypatch, files, expected_error
) -> None:
    """The server enforces the same limits, but its error arrives over SSE only
    after the user's message is already rendered and stuck in the transcript."""
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(
        chat_interface, "render_sandbox_interrupt", Mock(return_value=False)
    )
    stream_mock = Mock()
    error_mock = Mock()
    session_state = _SessionState(
        client=Mock(),
        messages=[],
        welcome_initialized=True,
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
        chat_input=Mock(return_value=SimpleNamespace(text="have a look", files=files)),
        container=lambda: _DummyContext(),
        error=error_mock,
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", Mock())
    monkeypatch.setattr(chat_interface, "render_message", Mock())
    monkeypatch.setattr(chat_interface, "render_starter_topics", Mock(return_value=None))
    monkeypatch.setattr(chat_interface, "stream_and_display_response", stream_mock)

    chat_interface.render_chat_interface()

    assert session_state.messages == []
    stream_mock.assert_not_called()
    assert expected_error in " ".join(str(call.args[0]) for call in error_mock.call_args_list)


def test_a_pending_interrupt_replaces_the_composer(monkeypatch) -> None:
    """A thread waiting on an answer offers the card instead of the input box.

    The card itself is core's, and every test above says there is no pending
    interrupt so that it stays out of the way. This is the one that says the
    page asks at all -- and that a "yes" ends the run, rather than drawing a
    composer underneath a question the user has not answered yet.
    """
    chat_interface = _import_chat_interface(monkeypatch)
    interrupt_mock = Mock(return_value=True)
    chat_input_mock = Mock(return_value=None)
    session_state = _SessionState(
        client=Mock(),
        messages=[ChatMessage(type="human", content="plot this")],
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        show_system_messages=False,
    )

    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(
            session_state=session_state,
            chat_message=lambda _role: _DummyContext(),
            empty=Mock(),
            chat_input=chat_input_mock,
            container=lambda: _DummyContext(),
            error=Mock(),
        ),
    )
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", Mock())
    monkeypatch.setattr(chat_interface, "render_message", Mock())
    monkeypatch.setattr(chat_interface, "render_starter_topics", Mock())
    monkeypatch.setattr(chat_interface, "render_sandbox_interrupt", interrupt_mock)
    monkeypatch.setattr(chat_interface, "stream_and_display_response", Mock())

    chat_interface.render_chat_interface()

    interrupt_mock.assert_called_once_with()
    chat_input_mock.assert_not_called()
