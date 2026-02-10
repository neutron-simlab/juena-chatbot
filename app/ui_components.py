"""
UI components for rendering messages in Streamlit.

This module provides functions for rendering chat messages and content
with consistent styling across the application.
"""
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import markdown

from juena.schema.server import ChatMessage

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Paths and assets
_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None

# Default styling color
DEFAULT_COLOR = "blue"


def render_header_with_logo() -> None:
    """Render a top header with logo if available."""
    if _logo_path and _logo_path.exists():
        st.image(str(_logo_path), width=200)
    st.title("JueNA Chatbot")


def markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML.
    
    Args:
        markdown_text: Markdown-formatted text string
        
    Returns:
        HTML string with markdown converted to HTML
    """
    if not markdown_text:
        return ""
    
    # Convert markdown to HTML
    html = markdown.markdown(str(markdown_text), extensions=['fenced_code', 'nl2br'])
    return html


def render_content(content: any, custom_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Render message content uniformly (JSON or markdown).
    
    Args:
        content: Message content (string, dict, list, or JSON string)
        custom_data: Optional custom data that may contain plot information
    """
    # Check for plot data in custom_data (if users add custom visualization)
    if custom_data:
        plot_data = custom_data.get("plot_data", {})
        if plot_data and PLOTLY_AVAILABLE:
            # Render plots if available
            for plot_key, plot_info in plot_data.items():
                plot_json = plot_info.get("plot_json")
                if plot_json:
                    try:
                        fig = go.Figure(plot_json)
                        with st.expander(f"📊 {plot_info.get('title', 'Plot')}", expanded=True):
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering plot: {str(e)}")
    
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
                render_content(message.content, custom_data=message.custom_data)
    
    elif message.type == "tool":
        # Tool messages
        with st.chat_message("assistant"):
            st.markdown("🔧 **Tool Result**")
            if message.content:
                render_content(message.content, custom_data=message.custom_data)
    
    elif message.type == "system":
        # System messages - plain text display
        with st.chat_message("assistant"):
            st.text(message.content)


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
