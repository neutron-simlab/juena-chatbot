"""
Chat interface for Streamlit app.
"""
from typing import Any

import streamlit as st
from juena.clients.client import AgentClient, AgentClientError
from juena.schema.server import ChatMessage
from juena.schema.upload_limits import (
    CHAT_INPUT_MAX_CHARS,
    CHAT_INPUT_MAX_UPLOAD_MB,
    TEXT_READABLE_FILE_TYPES,
    validate_attachments,
)
from app.starters import build_starter_prompts
from app.ui_components import (
    finalize_streaming_message,
    render_chat_input_styles,
    render_header_with_logo,
    render_message,
    render_streaming_token,
    reset_rendered_artifacts,
)
from juena.core.config import global_config


STARTER_PILLS_KEY = "starter_pills"

# Shown on the confirmation card once its buttons are spent, so a deactivated
# card still says why it went quiet.
DECISION_STATUS = {
    "approve": "Running in the sandbox…",
    "edit": "Running the edited command…",
    "reject": "Cancelled.",
}


def render_starter_topics() -> str | None:
    """Render starter topic chips for an empty chat.

    Returns the chosen question, or None when nothing was picked.

    Streamlit's ``st.chat_input`` cannot be pre-filled programmatically, so a
    chip sends its question rather than staging it in the composer. The question
    is still shown verbatim as the user's own message, so nothing is hidden.
    """
    starters = build_starter_prompts()
    if not starters:
        return None

    st.caption("Not sure where to start?")
    selected_label = st.pills(
        "Starter topics",
        options=[starter.label for starter in starters],
        selection_mode="single",
        default=None,
        key=f"{STARTER_PILLS_KEY}:{st.session_state.thread_id}",
        label_visibility="collapsed",
    )
    if not selected_label:
        return None

    for starter in starters:
        if starter.label == selected_label:
            if starter.requires_sandbox and not global_config.SANDBOX_ENABLED:
                st.info(
                    "Python execution is not active in this local environment. "
                    "The chip is shown for interface review only."
                )
                return None
            return starter.prompt
    return None


def _replace_message_by_id(messages: list, chunk: ChatMessage) -> bool:
    """Overwrite an already-held message carrying the same id as *chunk*.

    Returns whether a slot was found. Messages without an id cannot be
    correlated, so they always count as new.
    """
    message_id = getattr(chunk, "id", None)
    if not message_id:
        return False
    for index, existing in enumerate(messages):
        if isinstance(existing, ChatMessage) and existing.id == message_id:
            messages[index] = chunk
            return True
    return False


def process_stream_chunk(
    chunk,
    client: AgentClient,
    response_text: str,
    received_complete_message: bool,
    message_placeholder,
    messages: list,
    pending_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[str, bool]:
    """
    Process a single chunk from the stream and update UI.
    
    Args:
        chunk: Stream chunk (ChatMessage or token dict)
        client: AgentClient instance for helper methods
        response_text: Accumulated response text
        received_complete_message: Whether complete message was received
        message_placeholder: Streamlit placeholder
        messages: Message history list
        
    Returns:
        Tuple of (updated_response_text, updated_complete_flag)
    """
    if isinstance(chunk, ChatMessage):
        # Complete message received
        content_str = str(chunk.content) if chunk.content is not None else ""
        # Skip stray 'Start' messages
        if content_str.strip().lower() == "start":
            return response_text, received_complete_message
        
        received_complete_message = True
        if chunk.type == "ai" and pending_artifacts:
            stored = chunk.custom_data.get("artifacts") or []
            stored_by_id = {
                item.get("artifact_id"): item
                for item in stored
                if isinstance(item, dict) and item.get("artifact_id")
            }
            live_by_id = {
                item.get("artifact_id"): item
                for item in pending_artifacts
                if isinstance(item, dict) and item.get("artifact_id")
            }
            artifact_ids = list(stored_by_id) or list(live_by_id)
            chunk.custom_data["artifacts"] = [
                {**stored_by_id.get(artifact_id, {}), **live_by_id.get(artifact_id, {})}
                for artifact_id in artifact_ids
            ]
            pending_artifacts.clear()
        # Resuming an interrupt replays the node that was interrupted, so the
        # stream can re-send a message this session already holds. Correlate on
        # the id the server copies from LangChain and overwrite in place, or the
        # transcript grows a second copy and its artifacts are drawn twice in
        # one script run -- which Streamlit rejects as a duplicate widget key.
        replaced = _replace_message_by_id(messages, chunk)
        if not replaced:
            messages.append(chunk)
        
        if chunk.type == "ai":
            if chunk.custom_data.get("artifacts"):
                finalize_streaming_message(
                    message_placeholder,
                    chunk.content,
                    chunk.custom_data,
                )
            else:
                finalize_streaming_message(message_placeholder, chunk.content)
            response_text = chunk.content
        
        return response_text, received_complete_message
    
    elif client.is_token_message(chunk):
        # Token message - accumulate
        if not received_complete_message:
            token_content = client.get_token_content(chunk) or ""
            
            if token_content:
                response_text += token_content
                render_streaming_token(response_text, message_placeholder)
        
        return response_text, received_complete_message
    
    return response_text, received_complete_message


def is_status_chunk(chunk: Any) -> bool:
    """Whether a stream chunk reports agent activity rather than content."""
    return isinstance(chunk, dict) and chunk.get("type") in {"status", "sandbox_status"}


def is_approval_chunk(chunk: Any) -> bool:
    """Whether a stream chunk asks the user to confirm sandbox execution."""
    return isinstance(chunk, dict) and chunk.get("type") == "approval_required"


def is_interrupt_chunk(chunk: Any) -> bool:
    """Whether a stream chunk pauses the turn to wait on the user."""
    return isinstance(chunk, dict) and chunk.get("type") in {
        "approval_required",
        "clarification_required",
    }


def is_artifact_chunk(chunk: Any) -> bool:
    """Whether a stream chunk contains one generated plot."""
    return isinstance(chunk, dict) and chunk.get("type") == "artifact"


def apply_status_chunk(status, chunk: dict) -> None:
    """Reflect an activity event in the status container.

    Tool calls are written as lines inside the container so the transcript
    stays free of raw tool payloads.
    """
    phase = chunk.get("phase", "")
    label = chunk.get("label", "")

    if chunk.get("type") == "sandbox_status":
        if chunk.get("status") in {"completed", "timed_out", "failed"}:
            status.write(label)
        elif label:
            status.update(label=label)
        return

    if phase == "tool":
        tool = chunk.get("tool") or "tool"
        summary = chunk.get("summary") or ""
        status.write(f"🔧 `{tool}` — {summary}" if summary else f"🔧 `{tool}`")
        return

    if phase == "done":
        # First answer token: collapse the activity log and get out of the way.
        status.update(label="Done", state="complete", expanded=False)
        return

    if label:
        status.update(label=label)


def _normalize_chat_submission(submission: Any) -> tuple[str, list[Any]]:
    """Normalize Streamlit chat input into text plus attached files."""

    if submission is None:
        return "", []
    if isinstance(submission, str):
        return submission, []
    text = str(getattr(submission, "text", "") or "")
    files = list(getattr(submission, "files", []) or [])
    return text, files


def _attachment_metadata(attachments: list[Any]) -> list[dict[str, Any]]:
    """Describe attached files for display on the user's message.

    Mirrors the shape the server records on the staged human message, so a live
    turn and a reloaded one render identically.
    """

    # Only the name: the server records a character count once it has decoded the
    # file, and reporting raw bytes under the same key would mislabel it.
    return [
        {"name": str(getattr(attachment, "name", "attachment"))}
        for attachment in attachments
    ]


def _attachment_errors(attachments: list[Any]) -> list[str]:
    """Return reasons the selected files cannot be sent, if any."""

    return validate_attachments(
        [
            (str(getattr(attachment, "name", "attachment")), getattr(attachment, "size", None))
            for attachment in attachments
        ]
    )


def _chat_input_kwargs() -> dict[str, Any]:
    """Return keyword arguments for the main Streamlit chat input."""

    # A single agent handles every conversation, so attachments are always on.
    return {
        "placeholder": "Type your message here...",
        "max_chars": CHAT_INPUT_MAX_CHARS,
        "accept_file": "multiple",
        "max_upload_size": CHAT_INPUT_MAX_UPLOAD_MB,
        "file_type": TEXT_READABLE_FILE_TYPES,
    }


def _stream_and_display_chunks(
    chunks,
    message_placeholder,
    should_rerun: bool = True,
) -> None:
    """Consume new-turn or approval-resume chunks through one UI path."""
    initial_message_count = len(st.session_state.messages)
    response_text = ""
    received_complete_message = False
    pending_artifacts: list[dict[str, Any]] = []
    waiting_for_user = False

    try:
        with st.status("Thinking…", expanded=False) as status:
            for chunk in chunks:
                if is_status_chunk(chunk):
                    apply_status_chunk(status, chunk)
                    continue
                if is_artifact_chunk(chunk):
                    pending_artifacts.append(chunk)
                    continue
                if is_interrupt_chunk(chunk):
                    approvals = dict(st.session_state.get("pending_approvals") or {})
                    approvals[st.session_state.thread_id] = chunk
                    st.session_state.pending_approvals = approvals
                    waiting_for_user = True
                    status.update(
                        label=(
                            "Waiting for confirmation"
                            if is_approval_chunk(chunk)
                            else "Waiting for your answer"
                        ),
                        expanded=False,
                    )
                    continue

                response_text, received_complete_message = process_stream_chunk(
                    chunk,
                    st.session_state.client,
                    response_text,
                    received_complete_message,
                    message_placeholder,
                    st.session_state.messages,
                    pending_artifacts,
                )

            if not waiting_for_user:
                status.update(label="Done", state="complete", expanded=False)

        # Finalize message display
        if response_text and not received_complete_message:
            custom_data = {"artifacts": pending_artifacts} if pending_artifacts else {}
            if custom_data:
                finalize_streaming_message(message_placeholder, response_text, custom_data)
            else:
                finalize_streaming_message(message_placeholder, response_text)
            ai_message = ChatMessage(type="ai", content=response_text, custom_data=custom_data)
            st.session_state.messages.append(ai_message)
        elif (
            not response_text
            # A complete message has already been drawn into the placeholder by
            # process_stream_chunk. Drawing it again in the same run re-registers
            # its artifact download buttons under keys Streamlit still holds.
            and not received_complete_message
            and len(st.session_state.messages) > initial_message_count
        ):
            # If no response text accumulated, check for last message
            last_msg = st.session_state.messages[-1]
            if isinstance(last_msg, ChatMessage) and last_msg.type == "ai":
                if last_msg.custom_data.get("artifacts"):
                    finalize_streaming_message(
                        message_placeholder,
                        last_msg.content,
                        last_msg.custom_data,
                    )
                else:
                    finalize_streaming_message(message_placeholder, last_msg.content)

        if should_rerun:
            st.rerun()

    except AgentClientError as e:
        st.error(f"Error communicating with server: {e}")
        error_message = ChatMessage(type="ai", content=f"Error: {str(e)}")
        st.session_state.messages.append(error_message)
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        error_message = ChatMessage(type="ai", content=f"Unexpected error: {str(e)}")
        st.session_state.messages.append(error_message)


def stream_and_display_response(
    message: str,
    message_placeholder,
    should_rerun: bool = True,
    attachments: list[Any] | None = None,
) -> None:
    """Start a user turn and stream its activity and final answer."""
    chunks = st.session_state.client.stream(
        message=message,
        thread_id=st.session_state.thread_id,
        provider=st.session_state.selected_provider,
        model=st.session_state.selected_model,
        attachments=attachments,
    )
    _stream_and_display_chunks(chunks, message_placeholder, should_rerun)


def stream_and_display_resume(
    approval: dict[str, Any],
    decision: str | None,
    message_placeholder,
    *,
    edited_command: str | None = None,
    answer: str | None = None,
) -> None:
    """Resume the interrupted turn after a decision or an answer."""
    thread_id = st.session_state.thread_id
    interrupt_id = str(approval.get("interrupt_id") or "")
    approvals = dict(st.session_state.get("pending_approvals") or {})
    approvals.pop(thread_id, None)
    st.session_state.pending_approvals = approvals
    checked_threads = set(st.session_state.get("approval_checked_threads") or ())
    checked_threads.discard(thread_id)
    st.session_state.approval_checked_threads = checked_threads

    def resume_chunks():
        try:
            yield from st.session_state.client.resume_stream(
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                decision=decision,
                edited_command=edited_command,
                answer=answer,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
            )
        except Exception:
            current = dict(st.session_state.get("pending_approvals") or {})
            current[thread_id] = approval
            st.session_state.pending_approvals = current
            # The command is pending again, so its card has to accept a click
            # again on the next run.
            st.session_state.pop(_decision_key(thread_id, interrupt_id), None)
            raise

    _stream_and_display_chunks(resume_chunks(), message_placeholder)


def _decision_key(thread_id: str, interrupt_id: str) -> str:
    """Session-state key holding the decision already taken on a command."""
    return f"sandbox_decision:{thread_id}:{interrupt_id}"


def _render_approval_card(
    approval: dict[str, Any],
    command: str,
    *,
    allowed: set[str],
    keys: str,
    disabled: bool,
    show_editor: bool,
    status: str = "",
) -> tuple[str | None, str | None, bool]:
    """Draw the confirmation card once.

    Returns ``(decision, edited_command, edit_clicked)``. With ``disabled`` set
    every control is greyed out, which lets the caller redraw the card as an
    inert copy of itself the moment a decision is taken.
    """
    st.subheader("Allow this command?")
    description = str(approval.get("description") or "").strip()
    if description:
        st.write(description)
    st.code(command, language="bash")
    limits = approval.get("limits") or {}
    st.caption(
        " · ".join(
            part
            for part in (
                "No network",
                str(limits.get("cpu") or ""),
                str(limits.get("memory") or ""),
            )
            if part
        )
    )

    run_col, edit_col, cancel_col = st.columns(3)
    run_clicked = run_col.button(
        "Run",
        icon=":material/play_arrow:",
        type="primary",
        key=f"sandbox_run:{keys}",
        use_container_width=True,
        disabled=disabled or "approve" not in allowed,
    )
    edit_clicked = edit_col.button(
        "Edit",
        icon=":material/edit:",
        key=f"sandbox_edit_button:{keys}",
        use_container_width=True,
        disabled=disabled or "edit" not in allowed,
    )
    cancel_clicked = cancel_col.button(
        "Cancel",
        icon=":material/close:",
        key=f"sandbox_cancel:{keys}",
        use_container_width=True,
        disabled=disabled or "reject" not in allowed,
    )

    edited_command: str | None = None
    edit_confirmed = False
    if show_editor or edit_clicked:
        edited_command = st.text_area(
            "Command",
            value=command,
            key=f"sandbox_command:{keys}",
            height=140,
            disabled=disabled,
        )
        edit_confirmed = st.button(
            "Run edited command",
            icon=":material/play_arrow:",
            type="primary",
            key=f"sandbox_edit_confirm:{keys}",
            disabled=disabled,
        )

    if status:
        st.caption(status)

    decision = "approve" if run_clicked else "reject" if cancel_clicked else None
    if edit_confirmed:
        decision = "edit"
    return decision, edited_command, edit_clicked


def _render_question_card(
    clarification: dict[str, Any],
    *,
    keys: str,
    disabled: bool,
    status: str = "",
) -> str | None:
    """Draw the question card once, returning the user's answer when given.

    Mirrors `_render_approval_card`: with ``disabled`` set every control is
    greyed out so the caller can redraw an inert copy while the turn resumes.
    """
    asked_by = str(clarification.get("asked_by") or "").strip()
    st.subheader("A question for you")
    if asked_by:
        st.caption(f"asked by `{asked_by}`")
    st.write(str(clarification.get("question") or ""))

    options = [
        str(option)
        for option in (clarification.get("options") or [])
        if str(option).strip()
    ]
    chosen: str | None = None
    if options:
        for index, (column, option) in enumerate(zip(st.columns(len(options)), options)):
            if column.button(
                option,
                key=f"clarify_option:{keys}:{index}",
                use_container_width=True,
                disabled=disabled,
            ):
                chosen = option

    typed = st.text_area(
        "Other…" if options else "Your answer",
        key=f"clarify_text:{keys}",
        height=90,
        disabled=disabled,
        placeholder="Answer in your own words",
    )
    sent = st.button(
        "Send",
        icon=":material/send:",
        type="primary",
        key=f"clarify_send:{keys}",
        disabled=disabled,
    )
    if status:
        st.caption(status)

    if chosen:
        return chosen
    return typed.strip() if sent and typed and typed.strip() else None


def _render_pending_clarification(clarification: dict[str, Any]) -> bool:
    """Render the question card and resume the turn once it is answered."""
    interrupt_id = str(clarification.get("interrupt_id") or "")
    if not interrupt_id:
        return False
    answer_key = f"clarify_answer:{st.session_state.thread_id}:{interrupt_id}"
    answered = str(st.session_state.get(answer_key) or "")

    slot = st.empty()
    with slot.container(border=True):
        answer = _render_question_card(
            clarification,
            keys=interrupt_id,
            disabled=bool(answered),
            status="Answered. Continuing…" if answered else "",
        )
    if not answer:
        return False

    st.session_state[answer_key] = answer
    with slot.container(border=True):
        _render_question_card(
            clarification,
            keys=f"{interrupt_id}:answered",
            disabled=True,
            status=f"Answered: {answer}",
        )

    with st.chat_message("assistant"):
        stream_and_display_resume(clarification, None, st.empty(), answer=answer)
    return True


def render_pending_interrupt() -> bool:
    """Render controls for the current thread's pending command or question."""
    approvals = st.session_state.get("pending_approvals") or {}
    approval = approvals.get(st.session_state.thread_id)
    checked_threads = set(st.session_state.get("approval_checked_threads") or ())
    if (
        approval is None
        and st.session_state.messages
        and st.session_state.thread_id not in checked_threads
    ):
        try:
            approval = st.session_state.client.get_pending_interrupt(
                st.session_state.thread_id,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
            )
        except AgentClientError as exc:
            st.warning(f"Unable to restore what this chat was waiting for: {exc}")
        checked_threads.add(st.session_state.thread_id)
        st.session_state.approval_checked_threads = checked_threads
        if isinstance(approval, dict):
            approvals = dict(approvals)
            approvals[st.session_state.thread_id] = approval
            st.session_state.pending_approvals = approvals
    if not isinstance(approval, dict):
        return False
    if approval.get("type") == "clarification_required":
        return _render_pending_clarification(approval)
    if approval.get("type") != "approval_required":
        return False

    interrupt_id = str(approval.get("interrupt_id") or "")
    if not interrupt_id:
        return False
    command = str(approval.get("command") or "")
    allowed = set(approval.get("allowed_decisions") or ("approve", "edit", "reject"))
    edit_key = f"sandbox_edit:{st.session_state.thread_id}:{interrupt_id}"
    decision_key = _decision_key(st.session_state.thread_id, interrupt_id)
    decided = str(st.session_state.get(decision_key) or "")

    # The card lives in a placeholder so a decision can redraw it in place as a
    # dead control: Streamlit keeps the widgets on screen while the resumed turn
    # streams, and a second click would abort that stream mid-flight.
    slot = st.empty()
    with slot.container(border=True):
        decision, edited_command, edit_clicked = _render_approval_card(
            approval,
            command,
            allowed=allowed,
            keys=interrupt_id,
            disabled=bool(decided),
            show_editor=bool(st.session_state.get(edit_key, False)),
            status=DECISION_STATUS.get(decided, ""),
        )

    if edit_clicked:
        st.session_state[edit_key] = True
    if decision is None:
        return False

    st.session_state[decision_key] = decision
    decided_command = (
        edited_command if decision == "edit" and edited_command else command
    )
    with slot.container(border=True):
        _render_approval_card(
            approval,
            decided_command,
            allowed=allowed,
            keys=f"{interrupt_id}:decided",
            disabled=True,
            show_editor=False,
            status=DECISION_STATUS[decision],
        )

    with st.chat_message("assistant"):
        stream_and_display_resume(
            approval,
            decision,
            st.empty(),
            edited_command=edited_command if decision == "edit" else None,
        )
    return True


def render_chat_interface() -> None:
    """Render the main chat interface including header, message history, and input."""
    # Render header
    reset_rendered_artifacts()
    render_header_with_logo()
    render_chat_input_styles()
    
    # Display chat history (filter out system messages unless debug mode is enabled)
    for message in st.session_state.messages:
        render_message(message, show_system=st.session_state.show_system_messages)

    if render_pending_interrupt():
        return

    # On an empty thread the composer sits directly under the title, with the
    # starter chips beneath it, so the first thing the user sees is where to
    # type. `st.chat_input` pins itself to the bottom of the viewport only when
    # called at the top level of the script; inside a container it renders
    # inline. Once the thread has messages it returns to the bottom, which is
    # where a growing transcript expects it.
    starter_prompt: str | None = None
    if st.session_state.client and not st.session_state.messages:
        with st.container():
            submission = st.chat_input(**_chat_input_kwargs())
        starter_prompt = render_starter_topics()
    else:
        submission = st.chat_input(**_chat_input_kwargs())

    prompt, attachments = _normalize_chat_submission(submission)
    if submission is None and starter_prompt:
        prompt, attachments = starter_prompt, []

    if not prompt and not attachments:
        return
    if not st.session_state.client:
        st.error("The JüNA service is unavailable. Please try again later.")
        return

    # Refuse bad uploads before the message joins the transcript. The server
    # enforces the same limits, but by the time its error arrives over SSE the
    # user's message is already rendered and stuck in the history.
    if upload_errors := _attachment_errors(attachments):
        for error in upload_errors:
            st.error(error)
        return

    user_message = ChatMessage(
        type="human",
        content=prompt,
        custom_data=(
            {"attachments": _attachment_metadata(attachments)} if attachments else {}
        ),
    )
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_and_display_response(
            prompt,
            message_placeholder,
            attachments=attachments or None,
        )
