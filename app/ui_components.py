"""
UI components for rendering messages in Streamlit.

This module provides functions for rendering chat messages and content
with consistent styling across the application.
"""
import base64
import binascii
import html
import io
import json
import re
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
PLOT_PREVIEW_WIDTH_PX = 720
# Artifact ids already given a widget key during the current script run. A file
# can reach the page from both the stored transcript and the live stream, and
# Streamlit treats a repeated widget key as a fatal error.
RENDERED_ARTIFACTS_KEY = "rendered_artifact_keys"
# Emit the bitmap at this multiple of the displayed size. 3 covers every common
# device pixel ratio; the file stays a few kilobytes at this scale.
HEADER_LOGO_PIXEL_DENSITY = 3
_DATA_IMAGE_MARKER_RE = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,",
    flags=re.IGNORECASE,
)
_BASE64_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)
# Every assistant image in this product is rendered from a validated artifact, so
# a Markdown image pointing anywhere else is a dead link -- most often a sandbox:
# path the model invented for a file that was never delivered. Stored threads
# still contain them, so this runs on history as well as new answers.
_UNRENDERABLE_IMAGE_RE = re.compile(
    r"!\[[^\]\n]*\]\((?!\s*https?://)\s*[^)\n]*\)",
    flags=re.IGNORECASE,
)
_DEAD_SCHEME_LINK_RE = re.compile(
    r"\[([^\]\n]*)\]\(\s*(?:sandbox|file):[^)\n]*\)",
    flags=re.IGNORECASE,
)


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


def sanitize_assistant_content(content: Any) -> Any:
    """Remove image markup Streamlit cannot resolve, preserving surrounding text.

    Generated images are rendered from validated artifacts, so any image the
    model writes inline is dead: an older turn may carry a very large Base64
    data URI, which is slow to send and renders broken, and a turn whose file
    was never delivered may carry an invented `sandbox:` path.
    """
    if not isinstance(content, str):
        return content

    cleaned = content
    search_from = 0
    while match := _DATA_IMAGE_MARKER_RE.search(cleaned, search_from):
        start = match.start()
        markdown_start = cleaned.rfind("![", max(0, start - 512), start)
        if markdown_start >= 0 and re.fullmatch(
            r"!\[[^\]\n]*\]\(\s*",
            cleaned[markdown_start:start],
        ):
            start = markdown_start

        end = match.end()
        while end < len(cleaned) and cleaned[end] in _BASE64_CHARACTERS:
            end += 1
        if end < len(cleaned) and cleaned[end] == ")":
            end += 1
        cleaned = cleaned[:start] + cleaned[end:]
        search_from = start

    # The scan above stays because it handles unterminated multi-megabyte blobs,
    # which a regex bounded to one line cannot match.
    cleaned = _UNRENDERABLE_IMAGE_RE.sub("", cleaned)
    cleaned = _DEAD_SCHEME_LINK_RE.sub(r"\1", cleaned)

    return re.sub(r"\n{3,}", "\n\n", cleaned)


def render_content(content: Any) -> None:
    """
    Render message content uniformly (JSON or markdown).
    
    Args:
        content: Message content (string, dict, list, or JSON string)
    """
    if isinstance(content, (dict, list)):
        st.json(content)
    else:
        try:
            parsed = json.loads(str(content))
            st.json(parsed)
        except (json.JSONDecodeError, TypeError):
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


def _artifact_bytes(artifact: dict[str, Any]) -> bytes | None:
    """Resolve an artifact from a live SSE payload or its authenticated URL."""
    encoded = artifact.get("content_base64")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None

    artifact_id = str(artifact.get("artifact_id") or "").strip()
    client = getattr(st.session_state, "client", None)
    if not artifact_id or client is None:
        return None
    try:
        return client.get_artifact(artifact_id)
    except Exception:
        return None


@st.dialog("Plot preview", width="large", icon=":material/monitoring:")
def _render_plot_dialog(content: bytes, caption: str) -> None:
    """Show a generated PNG at the dialog's available width."""
    _render_png(content, caption, max_width="100%")


def _render_png(content: bytes, caption: str, *, max_width: str) -> None:
    """Render validated PNG bytes without a temporary Streamlit media URL."""
    encoded = base64.b64encode(content).decode("ascii")
    safe_caption = html.escape(caption, quote=True)
    st.html(
        f"""
        <figure style="margin:0;max-width:{max_width};width:100%;">
          <img
            src="data:image/png;base64,{encoded}"
            alt="{safe_caption}"
            style="display:block;max-width:100%;width:100%;height:auto;"
          >
          <figcaption style="margin-top:0.35rem;color:#6b7280;font-size:0.875rem;">
            {safe_caption}
          </figcaption>
        </figure>
        """
    )


def reset_rendered_artifacts() -> None:
    """Forget which artifacts have been drawn. Call once per script run."""
    st.session_state[RENDERED_ARTIFACTS_KEY] = set()


def _claim_artifact_key(artifact: Dict[str, Any], filename: str) -> Optional[str]:
    """Reserve the widget key suffix for *artifact* in this script run.

    Returns ``None`` when the same artifact has already been drawn: one file is
    worth showing once, and re-using its key is an error Streamlit raises rather
    than absorbs. Artifacts the server sent without an id cannot be recognised
    across renders, so they get a fresh suffix instead of being dropped.
    """
    drawn = st.session_state.get(RENDERED_ARTIFACTS_KEY)
    if not isinstance(drawn, set):
        drawn = set()
        st.session_state[RENDERED_ARTIFACTS_KEY] = drawn
    artifact_id = str(artifact.get("artifact_id") or "")
    key = artifact_id or f"{filename}#{len(drawn)}"
    if key in drawn:
        return None
    drawn.add(key)
    return key


def render_artifacts(custom_data: Optional[Dict[str, Any]]) -> None:
    """Render plots and authenticated downloads beneath the assistant response."""
    artifacts = (custom_data or {}).get("artifacts") or []
    if not isinstance(artifacts, list):
        return

    ordered = sorted(
        (item for item in artifacts if isinstance(item, dict)),
        key=lambda item: 0 if item.get("kind") == "image" else 1,
    )
    for artifact in ordered:
        content = _artifact_bytes(artifact)
        if not content:
            st.caption("This generated file is no longer available.")
            continue
        caption = str(artifact.get("caption") or artifact.get("filename") or "Plot")
        filename = str(artifact.get("filename") or "result.txt")
        mime_type = str(artifact.get("mime_type") or "application/octet-stream")
        key = _claim_artifact_key(artifact, filename)
        if key is None:
            continue
        if artifact.get("kind") == "image":
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                st.caption("This generated plot is no longer available.")
                continue
            _render_png(content, caption, max_width=f"{PLOT_PREVIEW_WIDTH_PX}px")
            with st.container(horizontal=True):
                if st.button(
                    "Expand plot",
                    icon=":material/open_in_full:",
                    help="Open a larger plot preview",
                    key=f"artifact-expand:{key}",
                ):
                    _render_plot_dialog(content, caption)
                st.download_button(
                    f"Download {filename}",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    icon=":material/download:",
                    key=f"artifact-download:{key}",
                )
        else:
            st.caption(caption)
            st.download_button(
                f"Download {filename}",
                data=content,
                file_name=filename,
                mime=mime_type,
                icon=":material/download:",
                key=f"artifact-download:{key}",
            )


def finalize_streaming_message(
    message_placeholder,
    content: Any,
    custom_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the final AI message using the same content renderer as history.

    Args:
        message_placeholder: Streamlit placeholder for the streamed message
        content: Final message content
    """
    with message_placeholder.container():
        if content:
            render_content(sanitize_assistant_content(content))
        render_artifacts(custom_data)


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
        labels.append(
            f"📎 `{name}` ({chars} chars)" if isinstance(chars, int) else f"📎 `{name}`"
        )

    if labels:
        st.caption(" · ".join(labels))


def render_message(message: ChatMessage, show_system: bool = False) -> None:
    """
    Render a chat message with consistent styling.
    
    Args:
        message: ChatMessage to display
        show_system: If True, display system messages. If False, skip them.
    """
    if message.type == "system" and not show_system:
        return

    if message.type == "human":
        with st.chat_message("user"):
            if content_contains_math_markup(message.content):
                render_content(message.content)
            else:
                st.write(message.content)
            render_attachment_chips(message.custom_data)
    
    elif message.type == "ai":
        with st.chat_message("assistant"):
            if message.content:
                render_content(sanitize_assistant_content(message.content))
            render_artifacts(message.custom_data)
    
    elif message.type == "tool":
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
        with st.chat_message("assistant"):
            st.text(message.content)

    elif message.type == "custom":
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
    message_placeholder.markdown(f"{sanitize_assistant_content(response_text)}▌")
