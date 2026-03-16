"""
UI components for rendering messages in Streamlit.

This module provides functions for rendering chat messages and content
with consistent styling across the application.
"""
import json
import streamlit as st
from typing import Any, Dict, Optional

from juena.schema.server import ChatMessage


def render_header_with_logo() -> None:
    """Render a top header with logo if available."""
    st.title("JüNA Chatbot")


def render_content(content: Any) -> None:
    """
    Render message content uniformly (JSON or markdown).
    
    Args:
        content: Message content (string, dict, list, or JSON string)
    """
    # Try to render JSON nicely if possible
    if isinstance(content, (dict, list)):
        st.json(content)
    else:
        try:
            # Try parsing as JSON string
            parsed = json.loads(str(content))
            st.json(parsed)
        except (json.JSONDecodeError, TypeError):
            # Render as markdown
            content_str = str(content) if content else ""
            if content_str.strip():
                st.markdown(content_str)


def should_collapse_tool_payload(custom_data: Optional[Dict[str, Any]]) -> bool:
    """Return True when tool payloads should render inside an expander."""
    custom_data = custom_data or {}
    display_mode = str(custom_data.get("display_mode", "")).strip().lower()
    if display_mode == "inline":
        return False
    if display_mode == "collapsed_by_default":
        return True
    # Legacy stored tool messages without custom_data should still collapse.
    return True


def finalize_streaming_message(message_placeholder, content: Any) -> None:
    """
    Render the final AI message using the same content renderer as history.

    Args:
        message_placeholder: Streamlit placeholder for the streamed message
        content: Final message content
    """
    with message_placeholder.container():
        if content:
            render_content(content)


def render_message(message: ChatMessage, show_system: bool = False) -> None:
    """
    Render a chat message with consistent styling.
    
    Args:
        message: ChatMessage to display
        show_system: If True, display system messages. If False, skip them.
    """
    # Skip system messages unless explicitly enabled
    if message.type == "system" and not show_system:
        return
    
    # Render based on message type
    if message.type == "human":
        # User messages - simple display
        with st.chat_message("user"):
            st.write(message.content)
    
    elif message.type == "ai":
        # AI messages
        with st.chat_message("assistant"):
            if message.content:
                render_content(message.content)
    
    elif message.type == "tool":
        # Tool messages
        with st.chat_message("assistant"):
            st.markdown("🔧 **Tool Result**")
            if message.tool_call_id:
                st.caption(f"Tool call ID: {message.tool_call_id}")
            if message.content:
                if should_collapse_tool_payload(message.custom_data):
                    with st.expander("View tool payload", expanded=False):
                        render_content(message.content)
                else:
                    render_content(message.content)
    
    elif message.type == "system":
        # System messages - plain text display
        with st.chat_message("assistant"):
            st.text(message.content)
    
    elif message.type == "custom":
        # Custom messages - render content
        with st.chat_message("assistant"):
            if message.content:
                render_content(message.content)


def render_streaming_token(
    response_text: str,
    message_placeholder
) -> None:
    """
    Render streaming text with cursor.
    
    Args:
        response_text: Accumulated response text so far
        message_placeholder: Streamlit placeholder for the message
    """
    # Display content with cursor
    message_placeholder.markdown(f"{response_text}▌")
