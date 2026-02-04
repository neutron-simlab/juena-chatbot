"""
UI components for rendering messages and badges in Streamlit.

This module provides functions for rendering chat messages, badges, and content
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

# Module color mapping for visual differentiation
MODULE_COLORS = {
    "default": "blue",      # Streamlit's primary blue
    "agent": "green",       # Success green
    "tool": "gray",         # Neutral gray for tools
}

# Module display names and icons
MODULE_INFO = {
    "default": {"name": "AI", "icon": ""},
    "agent": {"name": "AGENT", "icon": ""},
    "tool": {"name": "TOOL", "icon": "🔧"},
}

# Color palette for dynamic module assignment
COLOR_PALETTE = [
    "blue", "green", "orange", "violet", "red", "purple", 
    "pink", "yellow", "cyan", "teal", "indigo", "brown"
]


def get_module_color(module_name: str, dynamic_modules: Optional[Dict[str, Any]] = None) -> str:
    """
    Get color for a module, using dynamic info if available, otherwise fallback.
    
    Args:
        module_name: Module identifier
        dynamic_modules: Optional dictionary of dynamic module info
        
    Returns:
        Color string for the module
    """
    # First check hardcoded colors
    if module_name in MODULE_COLORS:
        return MODULE_COLORS[module_name]
    
    # If we have dynamic modules, assign color based on order
    if dynamic_modules and module_name in dynamic_modules:
        module_order = dynamic_modules[module_name].get("order", 999)
        color_index = (module_order - 1) % len(COLOR_PALETTE)
        return COLOR_PALETTE[color_index]
    
    # Fallback to default
    return MODULE_COLORS["default"]


def render_header_with_logo() -> None:
    """Render a top header with logo if available."""
    if _logo_path and _logo_path.exists():
        st.image(str(_logo_path), width=200)
    st.title("JueNA Chatbot")


def module_badge_html(module_display_name: str) -> str:
    """Render module badge as HTML."""
    return f'<strong>{module_display_name}</strong>'


def render_module_badge(module_name: str, dynamic_modules: Optional[Dict[str, Any]] = None) -> str:
    """
    Get formatted module badge HTML.
    
    Args:
        module_name: Module identifier
        dynamic_modules: Optional dictionary of dynamic module info from server
        
    Returns:
        Formatted badge HTML string
    """
    # Try to get display name from dynamic modules first
    if dynamic_modules and module_name in dynamic_modules:
        display_name = dynamic_modules[module_name].get("name", module_name.upper())
        return module_badge_html(display_name)
    
    # Fallback to hardcoded MODULE_INFO
    module_info = MODULE_INFO.get(module_name, MODULE_INFO["default"])
    return module_badge_html(module_info['name'])


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


def render_content(content: any, color: str, custom_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Render message content uniformly (JSON or markdown).
    
    Args:
        content: Message content (string, dict, list, or JSON string)
        color: Border color for styling
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
            # Render as markdown with color styling
            content_str = str(content) if content else ""
            if content_str.strip():
                # Convert markdown to HTML first, then wrap in styled div
                html_content = markdown_to_html(content_str)
                st.markdown(
                    f'<div style="border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;">{html_content}</div>',
                    unsafe_allow_html=True
                )


def render_message_header(badge_text: str, message_type: str) -> None:
    """
    Render consistent message header with badge and type indicator.
    
    Args:
        badge_text: Formatted module badge HTML
        message_type: Type of message ("ai" or "tool")
    """
    if message_type == "tool":
        st.markdown(
            f"{badge_text} | 🔧 <strong>Tool Result</strong>",
            unsafe_allow_html=True,
        )
    else:
        # AI message - just show badge
        st.markdown(badge_text, unsafe_allow_html=True)


def render_message(message: ChatMessage, show_system: bool = False) -> None:
    """
    Render a chat message uniformly with consistent styling across all message types.
    
    Args:
        message: ChatMessage to display
        show_system: If True, display system messages. If False, skip them.
    """
    # Skip system messages unless explicitly enabled
    if message.type == "system" and not show_system:
        return
    
    # Extract module information for color coding
    custom_data = message.custom_data or {}
    module_name = custom_data.get("module_name", "default")
    color = get_module_color(module_name)
    
    # Render based on message type
    if message.type == "human":
        # User messages - simple display
        with st.chat_message("user"):
            st.write(message.content)
    
    elif message.type == "ai":
        # AI messages - unified rendering with badges and content
        with st.chat_message("assistant"):
            badge_text = render_module_badge(module_name)
            
            # Render header (badge only for AI messages)
            render_message_header(badge_text, "ai")
            
            # Render content uniformly
            if message.content:
                render_content(message.content, color, custom_data=message.custom_data)
    
    elif message.type == "tool":
        # Tool messages - unified rendering same as AI messages
        with st.chat_message("assistant"):
            badge_text = render_module_badge(module_name)
            
            # Render header (badge + tool indicator)
            render_message_header(badge_text, "tool")
            
            # Render content uniformly
            if message.content:
                render_content(message.content, color, custom_data=message.custom_data)
    
    elif message.type == "system":
        # System messages - plain text display
        with st.chat_message("assistant"):
            st.text(message.content)


def render_streaming_token(
    module_name: str,
    response_text: str,
    message_placeholder
) -> None:
    """
    Render streaming text with module badge and color styling.
    
    Args:
        module_name: Module name for badge and color
        response_text: Accumulated response text so far
        message_placeholder: Streamlit placeholder for the message
    """
    color = get_module_color(module_name)
    badge_text = render_module_badge(module_name)
    
    # Convert markdown to HTML first, then wrap in styled div
    html_content = markdown_to_html(response_text)
    
    # Display badge and content with cursor
    message_placeholder.markdown(f"{badge_text}", unsafe_allow_html=True)
    message_placeholder.markdown(
        f'<div style="border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;">{html_content}▌</div>',
        unsafe_allow_html=True,
    )
