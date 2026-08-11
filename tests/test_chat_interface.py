"""Tests for chat streaming display behavior."""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from juena.clients.client import AgentClientError
from juena.schema.server import ChatMessage
from juena.schema.upload_limits import CHAT_INPUT_MAX_CHARS, CHAT_INPUT_MAX_UPLOAD_MB
from app.starters import StarterPrompt as chat_interface_starter


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


class _FakeStatus:
    """Stand-in for the st.status container used during streaming."""

    def __init__(self):
        self.labels: list[str] = []
        self.writes: list[str] = []
        self.state: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *, label=None, state=None, expanded=None):
        if label is not None:
            self.labels.append(label)
        if state is not None:
            self.state = state

    def write(self, text):
        self.writes.append(text)


class _ButtonColumn:
    def __init__(self, clicked_label: str, calls: list[tuple[str, dict]]) -> None:
        self.clicked_label = clicked_label
        self.calls = calls

    def button(self, label: str, **kwargs):
        self.calls.append((label, kwargs))
        # Streamlit never reports a click for a greyed-out button.
        if kwargs.get("disabled"):
            return False
        return label == self.clicked_label


class _FakePlaceholder:
    """Stand-in for st.empty(), which can be redrawn into more than once."""

    def __init__(self) -> None:
        self.container_calls = 0

    def container(self, **_kwargs):
        self.container_calls += 1
        return _DummyContext()


def _import_chat_interface(monkeypatch):
    fake_chat_storage = types.ModuleType("app.chat_storage")
    fake_chat_storage.get_chat_storage = Mock()

    monkeypatch.setitem(sys.modules, "app.chat_storage", fake_chat_storage)
    sys.modules.pop("app.chat_interface", None)
    return importlib.import_module("app.chat_interface")


def test_process_stream_chunk_ai_message_uses_finalize_renderer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
    message_placeholder = Mock()
    messages: list[ChatMessage] = []
    chunk = ChatMessage(type="ai", content="final response")

    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)

    response_text, received_complete = chat_interface.process_stream_chunk(
        chunk=chunk,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=message_placeholder,
        messages=messages,
    )

    assert response_text == "final response"
    assert received_complete is True
    assert messages == [chunk]
    finalize_mock.assert_called_once_with(message_placeholder, "final response")
    message_placeholder.markdown.assert_not_called()


def test_process_stream_chunk_replaces_a_replayed_message(monkeypatch) -> None:
    """Resuming an interrupt replays messages the transcript already holds."""
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", Mock())
    artifacts = [{"artifact_id": "plot-1", "filename": "plot.png"}]
    first = ChatMessage(
        type="ai",
        id="msg-1",
        content="here is the plot",
        custom_data={"artifacts": artifacts},
    )
    messages: list[ChatMessage] = [ChatMessage(type="human", content="plot"), first]
    replay = ChatMessage(
        type="ai",
        id="msg-1",
        content="here is the plot",
        custom_data={"artifacts": artifacts},
    )

    chat_interface.process_stream_chunk(
        chunk=replay,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=Mock(),
        messages=messages,
    )

    assert len(messages) == 2
    assert messages[1] is replay


def test_process_stream_chunk_appends_messages_without_an_id(monkeypatch) -> None:
    """Nothing to correlate on means the message has to count as new."""
    chat_interface = _import_chat_interface(monkeypatch)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", Mock())
    messages: list[ChatMessage] = [ChatMessage(type="ai", content="first")]

    chat_interface.process_stream_chunk(
        chunk=ChatMessage(type="ai", content="second"),
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=Mock(),
        messages=messages,
    )

    assert [message.content for message in messages] == ["first", "second"]


def test_process_stream_chunk_ai_message_preserves_raw_math_for_finalize_renderer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
    message_placeholder = Mock()
    messages: list[ChatMessage] = []
    chunk = ChatMessage(type="ai", content=r"Result: \(a^2+b^2=c^2\)")

    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)

    response_text, received_complete = chat_interface.process_stream_chunk(
        chunk=chunk,
        client=Mock(),
        response_text="",
        received_complete_message=False,
        message_placeholder=message_placeholder,
        messages=messages,
    )

    assert response_text == r"Result: \(a^2+b^2=c^2\)"
    assert received_complete is True
    assert messages == [chunk]
    finalize_mock.assert_called_once_with(message_placeholder, r"Result: \(a^2+b^2=c^2\)")


def test_stream_and_display_response_token_only_completion_uses_finalize_renderer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
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
        status=lambda *a, **k: _FakeStatus(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(chat_interface, "render_streaming_token", render_streaming_token_mock)

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


def test_stream_and_display_response_token_only_math_completion_keeps_raw_text(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    finalize_mock = Mock()
    render_streaming_token_mock = Mock()
    message_placeholder = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient(
            [
                {"type": "token", "content": r"Equation: \("},
                {"type": "token", "content": r"a+b\)"},
            ]
        ),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=Mock(),
        error=Mock(),
        status=lambda *a, **k: _FakeStatus(),
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(chat_interface, "render_streaming_token", render_streaming_token_mock)

    chat_interface.stream_and_display_response(
        "hello",
        message_placeholder,
        should_rerun=False,
    )

    assert [call.args for call in render_streaming_token_mock.call_args_list] == [
        (r"Equation: \(", message_placeholder),
        (r"Equation: \(a+b\)", message_placeholder),
    ]
    finalize_mock.assert_called_once_with(message_placeholder, r"Equation: \(a+b\)")
    assert session_state.messages[0].content == r"Equation: \(a+b\)"


def test_status_chunks_are_recognized(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)

    assert chat_interface.is_status_chunk({"type": "status", "phase": "thinking"}) is True
    assert chat_interface.is_status_chunk({"type": "token", "content": "x"}) is False
    assert chat_interface.is_status_chunk(ChatMessage(type="ai", content="x")) is False


def test_approval_and_artifact_chunks_are_recognized(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)

    assert chat_interface.is_approval_chunk({"type": "approval_required"}) is True
    assert chat_interface.is_artifact_chunk({"type": "artifact"}) is True
    assert chat_interface.is_artifact_chunk({"type": "token"}) is False


def test_pending_approval_offers_one_click_choices(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    calls: list[tuple[str, dict]] = []
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
        "description": "Create a plot",
        "limits": {"cpu": "2 vCPU", "memory": "4 GB"},
        "allowed_decisions": ["approve", "edit", "reject"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="plot")],
        pending_approvals={"thread-1": approval},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        code=Mock(),
        caption=Mock(),
        columns=lambda _count: [
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
        ],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(side_effect=[slot, Mock()]),
    )
    resume_mock = Mock()
    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "stream_and_display_resume", resume_mock)

    assert chat_interface.render_pending_interrupt() is True

    # The card is drawn twice into the same placeholder: live, then spent.
    assert slot.container_calls == 2
    assert [label for label, _kwargs in calls] == [
        "Run",
        "Edit",
        "Cancel",
        "Run",
        "Edit",
        "Cancel",
    ]
    assert [kwargs["disabled"] for _label, kwargs in calls[3:]] == [True, True, True]
    assert session_state["sandbox_decision:thread-1:interrupt-1"] == "approve"
    resume_mock.assert_called_once()
    assert resume_mock.call_args.args[:2] == (approval, "approve")


def test_pending_approval_ignores_clicks_once_a_decision_is_taken(monkeypatch) -> None:
    """A card whose decision is already in flight offers no live controls."""
    chat_interface = _import_chat_interface(monkeypatch)
    calls: list[tuple[str, dict]] = []
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
        "allowed_decisions": ["approve", "edit", "reject"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="plot")],
        pending_approvals={"thread-1": approval},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    session_state["sandbox_decision:thread-1:interrupt-1"] = "approve"
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        code=Mock(),
        caption=Mock(),
        columns=lambda _count: [
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
            _ButtonColumn("Run", calls),
        ],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(return_value=slot),
    )
    resume_mock = Mock()
    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "stream_and_display_resume", resume_mock)

    assert chat_interface.render_pending_interrupt() is False

    assert all(kwargs["disabled"] for _label, kwargs in calls)
    resume_mock.assert_not_called()


def test_pending_question_offers_suggestions_and_a_free_text_answer(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    calls: list[tuple[str, dict]] = []
    clarification = {
        "type": "clarification_required",
        "interrupt_id": "interrupt-2",
        "asked_by": "software-specialist",
        "question": "Which background subtraction?",
        "options": ["Solvent-only", "Empty cell"],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="reduce this")],
        pending_approvals={"thread-1": clarification},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    slot = _FakePlaceholder()
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        caption=Mock(),
        text_area=Mock(return_value=""),
        button=Mock(return_value=False),
        columns=lambda count: [
            _ButtonColumn("Empty cell", calls) for _ in range(count)
        ],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(side_effect=[slot, Mock()]),
    )
    resume_mock = Mock()
    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "stream_and_display_resume", resume_mock)

    assert chat_interface.render_pending_interrupt() is True

    # Live card, then a spent copy redrawn into the same placeholder.
    assert slot.container_calls == 2
    assert [label for label, _kwargs in calls[:2]] == ["Solvent-only", "Empty cell"]
    assert all(kwargs["disabled"] for _label, kwargs in calls[2:])
    assert session_state["clarify_answer:thread-1:interrupt-2"] == "Empty cell"
    resume_mock.assert_called_once()
    assert resume_mock.call_args.kwargs["answer"] == "Empty cell"
    assert resume_mock.call_args.args[1] is None


def test_a_question_waits_until_an_answer_is_actually_given(monkeypatch) -> None:
    """Rendering the card must not resume the turn on its own."""
    chat_interface = _import_chat_interface(monkeypatch)
    calls: list[tuple[str, dict]] = []
    clarification = {
        "type": "clarification_required",
        "interrupt_id": "interrupt-2",
        "asked_by": "juena",
        "question": "Which instrument?",
        "options": [],
    }
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[ChatMessage(type="human", content="explain")],
        pending_approvals={"thread-1": clarification},
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        container=lambda **_kwargs: _DummyContext(),
        subheader=Mock(),
        write=Mock(),
        caption=Mock(),
        text_area=Mock(return_value="   "),
        button=Mock(return_value=True),
        columns=lambda count: [_ButtonColumn("", calls) for _ in range(count)],
        chat_message=lambda _role: _DummyContext(),
        empty=Mock(return_value=_FakePlaceholder()),
    )
    resume_mock = Mock()
    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "stream_and_display_resume", resume_mock)

    # Send was clicked, but the box holds only whitespace.
    assert chat_interface.render_pending_interrupt() is False
    resume_mock.assert_not_called()


def test_failed_resume_reopens_the_command_for_another_click(monkeypatch) -> None:
    """A resume that never starts must leave a clickable card behind."""
    chat_interface = _import_chat_interface(monkeypatch)
    approval = {
        "type": "approval_required",
        "interrupt_id": "interrupt-1",
        "command": "python plot.py",
    }
    client = Mock()
    client.resume_stream.side_effect = AgentClientError("boom")
    session_state = _SessionState(
        thread_id="thread-1",
        messages=[],
        pending_approvals={"thread-1": approval},
        approval_checked_threads=set(),
        client=client,
        selected_provider="openai",
        selected_model="gpt-4o-mini",
    )
    session_state["sandbox_decision:thread-1:interrupt-1"] = "approve"
    monkeypatch.setattr(
        chat_interface,
        "st",
        SimpleNamespace(session_state=session_state),
    )

    def drain(chunks, *_args, **_kwargs):
        """Stand in for the display loop, which swallows client errors."""
        try:
            list(chunks)
        except AgentClientError:
            pass

    monkeypatch.setattr(chat_interface, "_stream_and_display_chunks", drain)

    chat_interface.stream_and_display_resume(approval, "approve", Mock())

    assert session_state["pending_approvals"] == {"thread-1": approval}
    assert "sandbox_decision:thread-1:interrupt-1" not in session_state


def test_tool_activity_is_logged_inside_the_status_container(monkeypatch) -> None:
    """Tool activity belongs in the collapsible status, not the transcript."""
    chat_interface = _import_chat_interface(monkeypatch)
    status = _FakeStatus()

    chat_interface.apply_status_chunk(
        status,
        {
            "type": "status",
            "phase": "tool",
            "label": "Ran search_code_semantic",
            "tool": "search_code_semantic",
            "summary": "12 hits",
        },
    )

    assert status.writes == ["🔧 `search_code_semantic` — 12 hits"]
    assert status.labels == []


def test_status_relabels_and_completes(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
    status = _FakeStatus()

    chat_interface.apply_status_chunk(
        status, {"type": "status", "phase": "researching", "label": "Researching…"}
    )
    assert status.labels == ["Researching…"]
    assert status.state is None

    # The first answer token collapses the activity log.
    chat_interface.apply_status_chunk(status, {"type": "status", "phase": "done", "label": ""})
    assert status.state == "complete"


def test_stream_routes_status_chunks_away_from_response_text(monkeypatch) -> None:
    """A status event must never be accumulated into the answer."""
    chat_interface = _import_chat_interface(monkeypatch)
    status = _FakeStatus()
    finalize_mock = Mock()
    session_state = SimpleNamespace(
        client=_FakeTokenClient(
            [
                {"type": "status", "phase": "researching", "label": "Researching…"},
                {"type": "status", "phase": "tool", "tool": "grep", "summary": "3 hits"},
                {"type": "token", "content": "Answer"},
            ]
        ),
        thread_id="thread-1",
        user_id="user-1",
        selected_provider="openai",
        selected_model="gpt-4o-mini",
        messages=[],
    )
    fake_st = SimpleNamespace(
        session_state=session_state,
        rerun=Mock(),
        error=Mock(),
        status=lambda *a, **k: status,
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "finalize_streaming_message", finalize_mock)
    monkeypatch.setattr(chat_interface, "render_streaming_token", Mock())

    chat_interface.stream_and_display_response("hi", Mock(), should_rerun=False)

    # Only the token text became the answer; the status events did not.
    assert len(session_state.messages) == 1
    assert session_state.messages[0].content == "Answer"
    assert "Researching…" in status.labels
    assert status.writes == ["🔧 `grep` — 3 hits"]


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


def test_render_chat_interface_uses_attachment_enabled_chat_input(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
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
    )

    monkeypatch.setattr(chat_interface, "st", fake_st)
    monkeypatch.setattr(chat_interface, "render_header_with_logo", Mock())
    monkeypatch.setattr(chat_interface, "render_chat_input_styles", render_chat_input_styles_mock)
    monkeypatch.setattr(chat_interface, "render_message", Mock())
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


def test_render_chat_interface_streams_attachments(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
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


def test_render_chat_interface_preserves_raw_math_in_user_message(monkeypatch) -> None:
    chat_interface = _import_chat_interface(monkeypatch)
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
