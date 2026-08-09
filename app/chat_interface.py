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
)
from juena.core.config import global_config


STARTER_PILLS_KEY = "starter_pills"


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
        # Backend handles deduplication, so we can always append
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
    waiting_for_approval = False

    try:
        with st.status("Thinking…", expanded=False) as status:
            for chunk in chunks:
                if is_status_chunk(chunk):
                    apply_status_chunk(status, chunk)
                    continue
                if is_artifact_chunk(chunk):
                    pending_artifacts.append(chunk)
                    continue
                if is_approval_chunk(chunk):
                    approvals = dict(st.session_state.get("pending_approvals") or {})
                    approvals[st.session_state.thread_id] = chunk
                    st.session_state.pending_approvals = approvals
                    waiting_for_approval = True
                    status.update(label="Waiting for confirmation", expanded=False)
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

            if not waiting_for_approval:
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
        elif not response_text and len(st.session_state.messages) > initial_message_count:
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
    decision: str,
    message_placeholder,
    *,
    edited_command: str | None = None,
) -> None:
    """Resume the interrupted turn after the user clicks a decision."""
    thread_id = st.session_state.thread_id
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
                interrupt_id=str(approval.get("interrupt_id") or ""),
                decision=decision,
                edited_command=edited_command,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
            )
        except Exception:
            current = dict(st.session_state.get("pending_approvals") or {})
            current[thread_id] = approval
            st.session_state.pending_approvals = current
            raise

    _stream_and_display_chunks(resume_chunks(), message_placeholder)


def render_pending_approval() -> bool:
    """Render click-first controls for the current thread's pending command."""
    approvals = st.session_state.get("pending_approvals") or {}
    approval = approvals.get(st.session_state.thread_id)
    checked_threads = set(st.session_state.get("approval_checked_threads") or ())
    if (
        approval is None
        and st.session_state.messages
        and st.session_state.thread_id not in checked_threads
    ):
        try:
            approval = st.session_state.client.get_pending_approval(
                st.session_state.thread_id,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
            )
        except AgentClientError as exc:
            st.warning(f"Unable to restore a pending command confirmation: {exc}")
        checked_threads.add(st.session_state.thread_id)
        st.session_state.approval_checked_threads = checked_threads
        if isinstance(approval, dict):
            approvals = dict(approvals)
            approvals[st.session_state.thread_id] = approval
            st.session_state.pending_approvals = approvals
    if not isinstance(approval, dict) or approval.get("type") != "approval_required":
        return False

    interrupt_id = str(approval.get("interrupt_id") or "")
    if not interrupt_id:
        return False
    command = str(approval.get("command") or "")
    allowed = set(approval.get("allowed_decisions") or ("approve", "edit", "reject"))
    edit_key = f"sandbox_edit:{st.session_state.thread_id}:{interrupt_id}"

    with st.container(border=True):
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
            key=f"sandbox_run:{interrupt_id}",
            use_container_width=True,
            disabled="approve" not in allowed,
        )
        edit_clicked = edit_col.button(
            "Edit",
            icon=":material/edit:",
            key=f"sandbox_edit_button:{interrupt_id}",
            use_container_width=True,
            disabled="edit" not in allowed,
        )
        cancel_clicked = cancel_col.button(
            "Cancel",
            icon=":material/close:",
            key=f"sandbox_cancel:{interrupt_id}",
            use_container_width=True,
            disabled="reject" not in allowed,
        )

        if edit_clicked:
            st.session_state[edit_key] = True

        edited_command: str | None = None
        edit_confirmed = False
        if st.session_state.get(edit_key, False):
            edited_command = st.text_area(
                "Command",
                value=command,
                key=f"sandbox_command:{interrupt_id}",
                height=140,
            )
            edit_confirmed = st.button(
                "Run edited command",
                icon=":material/play_arrow:",
                type="primary",
                key=f"sandbox_edit_confirm:{interrupt_id}",
            )

    decision = "approve" if run_clicked else "reject" if cancel_clicked else None
    if edit_confirmed:
        decision = "edit"
    if decision is None:
        return False

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
    render_header_with_logo()
    render_chat_input_styles()
    
    # Display chat history (filter out system messages unless debug mode is enabled)
    for message in st.session_state.messages:
        render_message(message, show_system=st.session_state.show_system_messages)

    if render_pending_approval():
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
