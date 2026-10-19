"""The JüNA-specific half of the Streamlit shell: the logo, the header, the CSS.

Everything that renders a `ChatMessage`, an artifact or a streamed token moved
to `juena_core.ui.components` (00-BOUNDARY.md, decision 7): those are shapes
core's server defines, so core is the one place that has to agree with them.

What stayed is what says which product this is. A logo is domain, not wiring.

The core renderers are re-exported here so the pages that use them keep one
import site, and so this module remains the answer to "where does the page get
its rendering from".
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import streamlit as st

from juena_core.ui.components import (
    PLOT_PREVIEW_WIDTH_PX,
    RENDERED_ARTIFACTS_KEY,
    finalize_streaming_message,
    render_artifacts,
    render_attachment_chips,
    render_content,
    render_message,
    render_streaming_token,
    reset_rendered_artifacts,
    sanitize_assistant_content,
    should_collapse_tool_payload,
)

__all__ = [
    # This application's
    "CHAT_INPUT_MIN_HEIGHT_REM",
    "CHAT_INPUT_TEXTAREA_MIN_HEIGHT_REM",
    "FILES_PANEL_KEY",
    "LOGO_PATH",
    "HEADER_LOGO_WIDTH_PX",
    "HEADER_LOGO_PIXEL_DENSITY",
    "logo_data_uri",
    "render_header_with_logo",
    "render_chat_input_styles",
    "render_files_panel_styles",
    # Core's, re-exported
    "PLOT_PREVIEW_WIDTH_PX",
    "RENDERED_ARTIFACTS_KEY",
    "finalize_streaming_message",
    "render_artifacts",
    "render_attachment_chips",
    "render_content",
    "render_message",
    "render_streaming_token",
    "reset_rendered_artifacts",
    "sanitize_assistant_content",
    "should_collapse_tool_payload",
]

CHAT_INPUT_MIN_HEIGHT_REM = 7.5
CHAT_INPUT_TEXTAREA_MIN_HEIGHT_REM = 5.5

# The container holding the file browser. Streamlit turns a container key into
# the CSS class ``st-key-<key>``, which is how the styles below find its column.
FILES_PANEL_KEY = "files_panel"

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


def render_files_panel_styles() -> None:
    """Keep the file browser's column in view while the chat scrolls.

    The column sticks just below Streamlit's top bar. ``align-self`` stops it
    stretching to the chat's height, which would leave it no room to move. A
    browser taller than the window scrolls inside itself instead of being cut
    off. Below 640px Streamlit stacks the columns, so the rule stays off there.

    ``--juena-files-height`` is the height core's file browser grows to with a
    file open, so the file fills the column instead of stopping at 340px.

    ``st.html`` with only a ``<style>`` tag takes no space on the page.
    """
    st.html(
        f"""
        <style>
        @media (min-width: 641px) {{
            div[data-testid="stColumn"]:has(.st-key-{FILES_PANEL_KEY}) {{
                position: sticky;
                top: 4rem;
                align-self: flex-start;
                max-height: calc(100vh - 5rem);
                overflow-y: auto;
                --juena-files-height: calc(100vh - 5rem);
            }}
        }}
        </style>
        """
    )
