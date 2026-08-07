"""
Chat interface for Streamlit app.
"""
from typing import Any

import streamlit as st
from juena.clients.client import AgentClient, AgentClientError
from juena.schema.server import ChatMessage
from juena.server.code_chat_inputs import (
    CHAT_INPUT_MAX_CHARS,
    CHAT_INPUT_MAX_UPLOAD_MB,
    TEXT_READABLE_FILE_TYPES,
)
from app.starters import build_starter_prompts
from app.ui_components import (
    finalize_streaming_message,
    render_chat_input_styles,
    render_header_with_logo,
    render_message,
    render_streaming_token,
)


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
            return starter.prompt
    return None


def process_stream_chunk(
    chunk,
    client: AgentClient,
    response_text: str,
    received_complete_message: bool,
    message_placeholder,
    messages: list,
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
        # Backend handles deduplication, so we can always append
        messages.append(chunk)
        
        if chunk.type == "ai":
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
    return isinstance(chunk, dict) and chunk.get("type") == "status"


def apply_status_chunk(status, chunk: dict) -> None:
    """Reflect an activity event in the status container.

    Tool calls are written as lines inside the container so the transcript
    stays free of raw tool payloads.
    """
    phase = chunk.get("phase", "")
    label = chunk.get("label", "")

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


def _format_user_submission(text: str, attachments: list[Any]) -> str:
    """Render the local user-message content shown in chat history."""

    attachment_names = [str(getattr(attachment, "name", "attachment")) for attachment in attachments]
    if text and attachment_names:
        return text + "\n\nAttachments:\n" + "\n".join(f"- {name}" for name in attachment_names)
    if attachment_names:
        return "Uploaded files:\n" + "\n".join(f"- {name}" for name in attachment_names)
    return text


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


def stream_and_display_response(
    message: str,
    message_placeholder,
    should_rerun: bool = True,
    attachments: list[Any] | None = None,
) -> None:
    """
    Stream a response from the agent and display it.

    Agent activity (thinking, subagent research, tool calls) is shown in a
    collapsible status container above the answer, so the transcript keeps only
    the conversation itself.

    Args:
        message: The message to send to the agent
        message_placeholder: Streamlit placeholder for streaming display
        should_rerun: Whether to rerun after streaming completes
        attachments: Files to stage for this turn
    """
    initial_message_count = len(st.session_state.messages)
    response_text = ""
    received_complete_message = False

    try:
        with st.status("Thinking…", expanded=False) as status:
            for chunk in st.session_state.client.stream(
                message=message,
                thread_id=st.session_state.thread_id,
                provider=st.session_state.selected_provider,
                model=st.session_state.selected_model,
                attachments=attachments,
            ):
                if is_status_chunk(chunk):
                    apply_status_chunk(status, chunk)
                    continue

                response_text, received_complete_message = process_stream_chunk(
                    chunk,
                    st.session_state.client,
                    response_text,
                    received_complete_message,
                    message_placeholder,
                    st.session_state.messages,
                )

            status.update(label="Done", state="complete", expanded=False)

        # Finalize message display
        if response_text and not received_complete_message:
            finalize_streaming_message(message_placeholder, response_text)
            ai_message = ChatMessage(type="ai", content=response_text)
            st.session_state.messages.append(ai_message)
        elif not response_text and len(st.session_state.messages) > initial_message_count:
            # If no response text accumulated, check for last message
            last_msg = st.session_state.messages[-1]
            if isinstance(last_msg, ChatMessage) and last_msg.type == "ai":
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


def render_chat_interface() -> None:
    """Render the main chat interface including header, message history, and input."""
    # Render header
    render_header_with_logo()
    render_chat_input_styles()
    
    # Display chat history (filter out system messages unless debug mode is enabled)
    for message in st.session_state.messages:
        render_message(message, show_system=st.session_state.show_system_messages)

    # Offer starter topics while the conversation is still empty.
    starter_prompt: str | None = None
    if st.session_state.client and not st.session_state.messages:
        starter_prompt = render_starter_topics()

    submission = st.chat_input(**_chat_input_kwargs())
    prompt, attachments = _normalize_chat_submission(submission)
    if submission is None and starter_prompt:
        prompt, attachments = starter_prompt, []

    if not prompt and not attachments:
        return
    if not st.session_state.client:
        st.error("The JüNA service is unavailable. Please try again later.")
        return

    user_message = ChatMessage(
        type="human",
        content=_format_user_submission(prompt, attachments),
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
