"""
Sidebar configuration UI for Streamlit app.
"""
from html import escape

import streamlit as st
from uuid import uuid4
from pathlib import Path
from datetime import datetime
import textwrap

from juena.clients.client import AgentAuthenticationError, AgentClientError
from juena.core.config import global_config
from juena_core.schema.llm_models import Provider
from juena_core.llms_providers import (
    get_available_providers,
    get_available_models,
    get_default_model,
    format_model_name,
)
from juena_core.ui.chat_storage import Chat, ChatStorage


_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None


def _format_chat_history_title(chat: Chat, max_chars: int = 22) -> tuple[str, str]:
    """Return a compact single-line label and the full title for tooltips."""
    title = " ".join(chat.title.split()).strip()
    if not title or title == "New Chat":
        title = f"Chat {chat.thread_id[:8]}"

    if len(title) <= max_chars:
        return title, title

    return textwrap.shorten(title, width=max_chars, placeholder="..."), title


def _logout_form_html(public_url: str) -> str:
    """Return a browser-submitted form so the HttpOnly cookie is cleared."""
    action = escape(f"{public_url.rstrip('/')}/auth/logout", quote=True)
    return f"""
    <form action="{action}" method="post" target="_self">
      <button type="submit" style="
        width: 100%;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 0.45rem 0.75rem;
        background: transparent;
        color: inherit;
        cursor: pointer;
        font: inherit;
      ">Sign out</button>
    </form>
    """


def _activate_chat(storage: ChatStorage, thread_id: str) -> None:
    """Load a saved chat into the current Streamlit session."""

    st.session_state.thread_id = thread_id
    st.session_state.messages = storage.load_messages(thread_id)
    st.session_state.welcome_initialized = True
    st.query_params["thread"] = thread_id


def _create_new_chat() -> None:
    """Start a blank conversation.

    Deliberately does not persist anything: the server creates the chat row
    when the first message is sent. Writing it here would leave an empty
    "Chat <id>" in the history for every new chat that is never used.
    """

    new_thread_id = str(uuid4())
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []
    st.session_state.welcome_initialized = False
    st.query_params["thread"] = new_thread_id


def _delete_chat_with_server_state(
    storage: ChatStorage,
    thread_id: str,
    *,
    is_current: bool,
) -> tuple[bool, str | None]:
    """Delete the user-owned chat and its server-side checkpoint state."""

    client = st.session_state.get("client")
    if client is None:
        return False, "Server connection is required to delete the persisted conversation."

    try:
        storage.delete_chat(thread_id)
    except AgentClientError as exc:
        return False, f"Failed to delete conversation state: {exc}"

    if is_current:
        remaining_chats = storage.list_chats(limit=1)
        if remaining_chats:
            _activate_chat(storage, remaining_chats[0].thread_id)
        else:
            _create_new_chat()

    return True, None


#: What the badge sends when the user asks for a finished result. The supervisor
#: collects through `check_research`; this only prompts the turn in which it can.
COLLECT_PROMPT = "Collect the finished background research."

#: How often the badge re-checks. Background research takes minutes, so this is
#: about the user not having to guess, not about being immediate.
_RESEARCH_POLL_SECONDS = 15


@st.fragment(run_every=_RESEARCH_POLL_SECONDS)
def _render_research_status() -> None:
    """Tell the user when background research is running, and when it is ready.

    A fragment rather than a full rerun: this polls on a timer, and reloading the
    whole page every fifteen seconds would interrupt whatever they are typing.
    """
    _render_research_status_content()


def _render_research_status_content() -> None:
    """Render one status observation without hiding polling failures."""
    client = st.session_state.get("client")
    thread_id = st.session_state.get("thread_id")
    if not client or not thread_id:
        return

    try:
        status = client.list_research(str(thread_id))
    except AgentAuthenticationError:
        # A fragment rerun does not execute streamlit_app.py's authentication
        # bootstrap. Return to a full run so an expired cookie reaches the
        # existing sign-in flow instead of looking like an empty job list.
        st.rerun(scope="app")
        return
    except AgentClientError:
        st.caption("⚠️ Background research status unavailable")
        return

    running = int(status.get("running") or 0)
    ready = int(status.get("ready") or 0)
    if not running and not ready:
        return

    st.subheader("Background research")
    if running:
        st.caption(f"⏳ {running} running")
    if ready:
        label = "1 result ready" if ready == 1 else f"{ready} results ready"
        if st.button(f"📥 {label}", width="stretch", key="collect-research"):
            st.session_state.queued_prompt = COLLECT_PROMPT
            # The chat interface lives outside this fragment, so the whole app
            # has to rerun for it to see the queued prompt.
            st.rerun(scope="app")
    st.divider()


def render_sidebar() -> None:
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        if _logo_path and _logo_path.exists():
            st.image(str(_logo_path), width='stretch')
        st.title("JüNA")

        st.subheader("Account")
        current_user = st.session_state.get("user") or {}
        account_label = current_user.get("display_name") or current_user.get("email")
        if account_label:
            st.caption(str(account_label))
        st.html(_logout_form_html(global_config.JUENA_PUBLIC_URL))

        st.divider()

        _render_research_status()

        st.subheader("LLM Configuration")

        available_providers = get_available_providers()
        provider_options = [p.value for p in Provider if available_providers.get(p.value, False)]

        if not provider_options:
            st.error("❌ No LLM providers are configured. Please configure at least one provider (OpenAI, Blablador, etc.) with valid API keys in your .env file.")
            return

        if st.session_state.selected_provider not in provider_options:
            st.session_state.selected_provider = provider_options[0]
            st.session_state.selected_model = get_default_model(st.session_state.selected_provider)
            st.warning(f"⚠️ Previously selected provider is unavailable. Auto-selected: **{st.session_state.selected_provider}**")
        
        selected_provider = st.radio(
            "Provider",
            options=provider_options,
            index=provider_options.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_options else 0,
            help="Select the LLM provider to use"
        )
        
        if selected_provider != st.session_state.selected_provider:
            st.session_state.selected_provider = selected_provider
            st.session_state.selected_model = get_default_model(selected_provider)
            st.info(f"Switched to **{selected_provider}**. Next message will use the new model.")

        model_options = get_available_models(st.session_state.selected_provider)

        if st.session_state.selected_model not in model_options:
            default = get_default_model(st.session_state.selected_provider)
            st.session_state.selected_model = default if default in model_options else (model_options[0] if model_options else "")

        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            format_func=lambda m: format_model_name(st.session_state.selected_provider, m),
            help="Select the model to use"
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.info(f"Model changed to **{selected_model}**. Next message will use the new model.")

        show_system = st.checkbox(
            "Show system messages",
            value=st.session_state.show_system_messages,
            help="Display system/debug messages in the chat"
        )
        if show_system != st.session_state.show_system_messages:
            st.session_state.show_system_messages = show_system
            st.rerun()

        st.divider()

        st.subheader("Chat History")
        storage = st.session_state.chat_storage

        if st.button("➕ New Chat", help="Start a new conversation", use_container_width=True):
            _create_new_chat()
            st.rerun()

        chats = storage.list_chats(limit=20)

        editing_thread_id = st.session_state.get("editing_thread_id")
        if editing_thread_id:
            editing_chat = storage.get_chat(editing_thread_id)
            if editing_chat:
                st.caption("Rename conversation:")
                new_title = st.text_input(
                    "New name",
                    value=editing_chat.title if editing_chat.title != "New Chat" else "",
                    key="rename_input",
                    placeholder="Enter chat name...",
                    label_visibility="collapsed"
                )
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("Save", key="save_rename", use_container_width=True):
                        if new_title.strip():
                            editing_chat.title = new_title.strip()
                        else:
                            editing_chat.title = "New Chat"
                        editing_chat.updated_at = datetime.now()
                        storage.upsert_chat(editing_chat)
                        del st.session_state.editing_thread_id
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key="cancel_rename", use_container_width=True):
                        del st.session_state.editing_thread_id
                        st.rerun()
                st.divider()
        
        if chats:
            st.caption("Recent conversations:")
            for chat in chats:
                is_current = chat.thread_id == st.session_state.thread_id
                
                display_title, full_title = _format_chat_history_title(chat)
                
                # Give the title most of the width and collapse actions into a popover.
                select_col, actions_col = st.columns([6, 1], gap="small")
                
                with select_col:
                    button_type = "primary" if is_current else "secondary"
                    if st.button(
                        display_title,
                        key=f"chat_{chat.thread_id}",
                        use_container_width=True,
                        type=button_type,
                        disabled=is_current,
                        help=full_title,
                    ):
                        _activate_chat(storage, chat.thread_id)
                        st.rerun()
                
                with actions_col:
                    with st.popover("⋯"):
                        st.caption(full_title)
                        if is_current:
                            st.caption("Current conversation")

                        if st.button(
                            "Rename",
                            key=f"write_{chat.thread_id}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_thread_id = chat.thread_id
                            st.rerun()

                        if st.button(
                            "Delete",
                            key=f"delete_{chat.thread_id}",
                            use_container_width=True,
                        ):
                            deleted, error_message = _delete_chat_with_server_state(
                                storage,
                                chat.thread_id,
                                is_current=is_current,
                            )
                            if deleted:
                                st.rerun()
                            if error_message:
                                st.error(error_message)
        else:
            st.caption("No saved conversations yet.")
        
        st.divider()

        st.subheader("About")
        st.info(
            """
            **JüNA Chatbot** is a production-ready neutron science chatbot
            built with LangGraph, FastAPI, and Streamlit.
            
            It supports custom LangGraph agents to power
            tailored neutron science chatbot experiences.
            """
        )
