"""
UI components for rendering messages in Streamlit.

This module provides functions for rendering chat messages and content
with consistent styling across the application.
"""
import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from app.math_rendering import content_contains_math_markup, normalize_math_markdown
from juena.schema.server import ChatMessage

CHAT_INPUT_MIN_HEIGHT_REM = 7.5
CHAT_INPUT_TEXTAREA_MIN_HEIGHT_REM = 5.5

# The JüNA mark, shown above the chat title. The source is square and has an
# alpha channel, so it sits directly on the themed background with no plate.
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
HEADER_LOGO_WIDTH_PX = 88
# Emit the bitmap at this multiple of the displayed size. 3 covers every common
# device pixel ratio; the file stays a few kilobytes at this scale.
HEADER_LOGO_PIXEL_DENSITY = 3


@lru_cache(maxsize=4)
def logo_data_uri(width_px: int, density: int) -> str | None:
    """Return the JüNA mark as a base64 PNG data URI, sized for high-DPI screens.

    Not ``st.image(path, width=N)``: that resamples server-side to exactly N
    pixels with a bilinear filter, which both aliases badly on a large
    reduction and leaves the browser upscaling an 88px bitmap to 176 device
    pixels on a 2x display. The result looked blurry. Emitting the image at
    ``density`` times the displayed size and letting CSS scale it down keeps it
    sharp at any device pixel ratio, and Lanczos resampling suits the ratio.

    Returns ``None`` when the asset is absent, so a checkout without it still
    renders a usable header.
    """
    if not LOGO_PATH.is_file():
        return None

    from PIL import Image

    with Image.open(LOGO_PATH) as image:
        source_width, source_height = image.size
        target_width = width_px * density
        target_height = max(1, round(source_height * target_width / source_width))
        resized = image.convert("RGBA").resize(
            (target_width, target_height),
            resample=Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG", optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_header_with_logo() -> None:
    """Render the JüNA mark above the chat title."""
    data_uri = logo_data_uri(HEADER_LOGO_WIDTH_PX, HEADER_LOGO_PIXEL_DENSITY)
    if data_uri:
        st.html(
            f'<img src="{data_uri}" alt="JüNA" '
            f'style="width:{HEADER_LOGO_WIDTH_PX}px;height:auto;display:block;">'
        )
    st.title("JüNA Chatbot")


def render_chat_input_styles() -> None:
    """Inject lightweight CSS tweaks for a taller chat composer."""
    st.markdown(
        f"""
        <style>
        div[data-testid="stChatInput"] textarea {{
            min-height: {CHAT_INPUT_TEXTAREA_MIN_HEIGHT_REM}rem !important;
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
            line-height: 1.5;
        }}

        div[data-testid="stChatInput"] > div {{
            min-height: {CHAT_INPUT_MIN_HEIGHT_REM}rem;
            align-items: stretch;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
                st.markdown(normalize_math_markdown(content_str))


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


def render_attachment_chips(custom_data: Optional[Dict[str, Any]]) -> None:
    """Show the files attached to a user message, if any.

    The server records the same ``attachments`` key on the staged human message,
    so a reloaded chat shows the uploads it was sent with instead of losing every
    trace of them.
    """
    attachments = (custom_data or {}).get("attachments") or []
    if not isinstance(attachments, list):
        return

    labels: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        name = str(attachment.get("name") or "").strip()
        if not name:
            continue
        chars = attachment.get("chars")
        labels.append(f"📎 `{name}` ({chars} chars)" if isinstance(chars, int) else f"📎 `{name}`")

    if labels:
        st.caption(" · ".join(labels))


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
            if content_contains_math_markup(message.content):
                render_content(message.content)
            else:
                st.write(message.content)
            render_attachment_chips(message.custom_data)
    
    elif message.type == "ai":
        # AI messages
        with st.chat_message("assistant"):
            if message.content:
                render_content(message.content)
    
    elif message.type == "tool":
        # Tool messages
        with st.chat_message("assistant"):
            tool_name = str((message.custom_data or {}).get("tool_name") or "").strip()
            label = f"Tool Result · `{tool_name}`" if tool_name else "Tool Result"
            st.markdown(f"🔧 **{label}**")
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
