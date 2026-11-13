"""The JüNA chat page.

The stream loop, the chunk classifiers and the approval and question cards live
in `juena_core.ui.streaming`, because they implement the SSE (server-sent
events, the live reply stream) contract that core's server emits.

This module holds the layout, the starter chips, the upload rules the composer
enforces before a message joins the transcript, and the strings that describe
this deployment's sandbox to the person approving a command.
"""

from typing import Any
from uuid import uuid4

import streamlit as st

from juena.clients.client import AgentClientError
from juena_core.schema.server import ChatMessage
from juena_core.schema.upload_limits import (
    CHAT_INPUT_MAX_CHARS,
    validate_attachments,
)
from juena.uploads import UPLOAD_POLICY
from juena_core.ui.streaming import (
    ApprovalCard,
    render_pending_interrupt,
    render_running_response,
    update_instruction_messages,
)

from app.starters import build_starter_prompts
from app.ui_components import (
    render_chat_input_styles,
    render_header_with_logo,
    render_message,
    reset_rendered_artifacts,
)
from juena.core.config import global_config
from juena_core.ui.run_stream import get_stream_controller, start_response


STARTER_PILLS_KEY = "starter_pills"

# Shown on the confirmation card once its buttons are spent, so a deactivated
# card still says why it went quiet. Kept here, not in core, because "in the
# sandbox" is true only of this deployment.
DECISION_STATUS = {
    "approve": "Running in the sandbox…",
    "edit": "Running the edited command…",
    "reject": "Cancelled.",
}


def _sandbox_limits_caption(limits: Any) -> str:
    """Describe what the approved command will and will not be able to do.

    Every clause is a fact about this deployment's Podman sandbox (`network:
    none` and the CPU and memory limits the worker enforces), so the
    application says it, not core.
    """

    limits = limits or {}
    return " · ".join(
        part
        for part in (
            "No network",
            str(limits.get("cpu") or ""),
            str(limits.get("memory") or ""),
        )
        if part
    )


def _render_sandbox_status(status: Any, chunk: dict[str, Any]) -> None:
    """Show a `sandbox_status` event, which core forwards but does not read.

    A run that failed or timed out is written as a line, so the log says why
    its "Run" step failed; a successful run needs no line, because the step
    already says so. Anything else relabels the container in place, because a
    queued job can report progress many times before it runs.
    """

    label = chunk.get("label", "")
    if chunk.get("status") in {"timed_out", "failed"}:
        status.write(label)
    elif label:
        status.update(label=label)


#: The interrupt kind this deployment registers (`register_sandbox_interrupt`)
#: and composes into `create_app(resume_input=SandboxResumeInput)`. The card
#: resumes with this discriminator, so it must be that same name.
SANDBOX_APPROVAL_CARD = ApprovalCard(
    kind="execute_approval",
    heading="Allow this command?",
    decision_status=DECISION_STATUS,
    format_limits=_sandbox_limits_caption,
)

#: `sandbox_status` reaches the UI only because `create_app` allowlists it in
#: its `StreamPolicy`; this is the other half, the renderer for it.
SANDBOX_STATUS_RENDERERS = {"sandbox_status": _render_sandbox_status}


def render_sandbox_interrupt() -> bool:
    """Render this thread's pending approval or question, if there is one.

    Returns:
        Whether a card was drawn, so the page can skip its composer.
    """

    return render_pending_interrupt(
        approval=SANDBOX_APPROVAL_CARD,
        custom_status=SANDBOX_STATUS_RENDERERS,
    )


def render_starter_topics() -> str | None:
    """Render starter topic chips for an empty chat.

    Streamlit's ``st.chat_input`` cannot be pre-filled programmatically, so a
    chip sends its question rather than staging it in the composer. The question
    is still shown verbatim as the user's own message, so nothing is hidden.

    Returns:
        The chosen question, or None when nothing was picked.
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
        ],
        upload_policy=UPLOAD_POLICY,
    )


def _chat_input_kwargs() -> dict[str, Any]:
    """Return keyword arguments for the main Streamlit chat input."""

    if get_stream_controller().active:
        return {
            "placeholder": "Guide this response…", "key": "chat_prompt",
            "max_chars": CHAT_INPUT_MAX_CHARS, "accept_file": False,
            "disabled": get_stream_controller().run.get("state") != "running",
        }
    return {
        "placeholder": "Type your message here...",
        "key": "chat_prompt",
        "max_chars": CHAT_INPUT_MAX_CHARS,
        "accept_file": "multiple",
        "max_upload_size": UPLOAD_POLICY.max_binary_bytes // 1024**2,
        "file_type": sorted(suffix.lstrip(".") for suffix in UPLOAD_POLICY.suffixes),
    }

def render_chat_interface() -> None:
    """Render the main chat interface including header, message history, and input."""
    reset_rendered_artifacts()
    render_header_with_logo()
    render_chat_input_styles()
    controller = get_stream_controller()
    if controller.thread_id not in (None, st.session_state.thread_id):
        controller.close()
        controller.thread = None

    # Made before the composer so the turn it submits can be drawn here too.
    transcript = st.container()
    with transcript:
        for message in st.session_state.messages:
            render_message(message, show_system=st.session_state.show_system_messages)

    if not st.session_state.get("user", {}).get("chat_ready", False):
        from app.profile_page import open_profile, render_blablador_guidance
        render_blablador_guidance()
        st.button("Add Blablador key", on_click=open_profile, type="primary")
        return

    running = controller.active
    render_running_response(custom_status=SANDBOX_STATUS_RENDERERS)
    if not running and render_sandbox_interrupt():
        return
    if running:
        st.caption("Messages sent now guide the current response. Use Stop to cancel it.")
    if error := st.session_state.pop("steering_error", None):
        st.error(error)

    # The composer stays inside the chat column, below its transcript. On an
    # empty thread the starter chips sit beneath it. Inside a column
    # `st.chat_input` is drawn where it is called, not pinned to the bottom of
    # the window, so the turn it returns goes into `transcript` above it;
    # drawn here it would sit under the composer until the post-stream rerun.
    mention = st.session_state.pop("filesystem_mention", None)
    if mention and mention["thread_id"] == st.session_state.thread_id:
        st.session_state["chat_prompt"] = f"Please look at `{mention['path']}`: "
    if draft := st.session_state.pop("steering_draft", None):
        st.session_state["chat_prompt"] = draft["text"]
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
    # The sidebar's background-research badge queues a prompt rather than
    # sending one: it runs in a fragment, and only this function owns the
    # transcript. Popped so a later rerun does not send it twice.
    if submission is None and not starter_prompt and not running:
        if queued := st.session_state.pop("queued_prompt", None):
            prompt, attachments = str(queued), []

    if not prompt and not attachments:
        return
    if not st.session_state.client:
        st.error("The JüNA service is unavailable. Please try again later.")
        return

    if running:
        prior = st.session_state.get("steering_retry") or {}
        instruction_id = prior.get("id") if prior.get("text") == prompt and prior.get("run_id") == controller.run.get("run_id") else str(uuid4())
        try:
            snapshot = st.session_state.client.steer_run(
                st.session_state.thread_id, controller.run["run_id"], instruction_id, prompt,
            )
        except AgentClientError as exc:
            st.session_state.steering_retry = {"id": instruction_id, "text": prompt, "run_id": controller.run.get("run_id")}
            st.session_state.steering_draft = {"text": prompt}
            st.session_state.steering_error = str(exc)
            st.rerun()
            return
        controller.run = snapshot
        update_instruction_messages(snapshot)
        st.session_state.pop("steering_retry", None)
        st.rerun()
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
    with transcript:
        render_message(user_message)

    start_response(prompt, attachments=attachments or None)
