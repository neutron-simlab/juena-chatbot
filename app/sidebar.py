"""
Sidebar configuration UI for Streamlit app.
"""
import streamlit as st
from uuid import uuid4
from pathlib import Path
from datetime import datetime
import textwrap

import httpx

from juena.clients.client import AgentClientError
from juena.schema.llm_models import Provider
from juena.core.llms_providers import (
    get_available_providers,
    get_available_models,
    get_default_model,
    format_model_name,
)
from app.chat_storage import get_chat_storage, Chat
from app.file_management import check_server_health, initialize_client


def _fetch_agents(server_url: str) -> tuple[list[str], str]:
    """Fetch registered agents from the backend.  Returns (agent_list, default)."""
    try:
        resp = httpx.get(f"{server_url}/agents", timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("agents", []), data.get("default", "")
    except Exception:
        return [], ""

# Paths and assets
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


def _delete_chat_with_server_state(storage, thread_id: str, *, is_current: bool) -> tuple[bool, str | None]:
    """Delete both the server-side thread state and the local chat history."""

    client = st.session_state.get("client")
    if client is None or not st.session_state.get("server_connected", False):
        return False, "Server connection is required to delete the persisted conversation."

    try:
        client.delete_thread(thread_id)
    except AgentClientError as exc:
        return False, f"Failed to delete conversation state: {exc}"

    storage.delete_chat(thread_id)

    if is_current:
        new_thread_id = str(uuid4())
        storage.upsert_chat(Chat(thread_id=new_thread_id))
        st.session_state.thread_id = new_thread_id
        st.session_state.messages = []
        st.session_state.welcome_initialized = False

    return True, None


def render_sidebar() -> None:
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        if _logo_path and _logo_path.exists():
            st.image(str(_logo_path), width='stretch')
        st.title("JüNA")
        
        # 1. Server Configuration
        st.subheader("Server")
        
        # Server URL input
        server_url = st.text_input(
            "Server URL",
            value=st.session_state.server_url,
            help="URL of the backend API server",
            key="server_url_input"
        )
        
        # Update server URL if changed
        if server_url != st.session_state.server_url:
            st.session_state.server_url = server_url
            st.session_state._health_checked = False  # Force re-check
        
        # Connection status and recheck button
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.session_state.server_connected:
                st.success("Connected")
            else:
                st.error("Disconnected")
        with col2:
            if st.button("Check", help="Recheck server connection"):
                st.session_state.server_connected = check_server_health(st.session_state.server_url)
                if st.session_state.server_connected:
                    st.session_state.client = initialize_client(st.session_state.server_url)
                    if st.session_state.client is None:
                        st.session_state.server_connected = False
                else:
                    st.session_state.client = None
                st.rerun()
        
        st.divider()

        # 2. Agent Selection
        st.subheader("Agent")

        if st.session_state.server_connected:
            agent_list, default_agent = _fetch_agents(st.session_state.server_url)
        else:
            agent_list, default_agent = [], ""

        if not agent_list:
            agent_list = [st.session_state.get("selected_agent", "react_agent")]

        if "selected_agent" not in st.session_state:
            st.session_state.selected_agent = default_agent or agent_list[0]

        if st.session_state.selected_agent not in agent_list:
            st.session_state.selected_agent = agent_list[0]

        selected_agent = st.selectbox(
            "Active agent",
            options=agent_list,
            index=agent_list.index(st.session_state.selected_agent),
            help="Choose the LangGraph agent to handle messages",
        )

        if selected_agent != st.session_state.selected_agent:
            st.session_state.selected_agent = selected_agent
            if st.session_state.client:
                st.session_state.client.update_agent(selected_agent)
            st.info(f"Switched to agent **{selected_agent}**.")

        if st.session_state.client and st.session_state.client.agent != st.session_state.selected_agent:
            st.session_state.client.update_agent(st.session_state.selected_agent)

        st.divider()

        # 3. LLM Configuration
        st.subheader("LLM Configuration")
        
        # Get available providers dynamically
        available_providers = get_available_providers()
        provider_options = [p.value for p in Provider if available_providers.get(p.value, False)]
        
        # Handle edge case: No providers available
        if not provider_options:
            st.error("❌ No LLM providers are configured. Please configure at least one provider (OpenAI, Blablador, etc.) with valid API keys in your .env file.")
            return
        
        # Auto-select first available provider if current selection is unavailable
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
        
        # Handle provider change - use registry to get default model
        if selected_provider != st.session_state.selected_provider:
            st.session_state.selected_provider = selected_provider
            st.session_state.selected_model = get_default_model(selected_provider)
            st.info(f"Switched to **{selected_provider}**. Next message will use the new model.")
        
        # Model selector - generic for all providers
        model_options = get_available_models(st.session_state.selected_provider)
        
        # Ensure selected model is valid for current provider
        if st.session_state.selected_model not in model_options:
            default = get_default_model(st.session_state.selected_provider)
            st.session_state.selected_model = default if default in model_options else (model_options[0] if model_options else "")
        
        # Create model selector with provider-specific formatting
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            format_func=lambda m: format_model_name(st.session_state.selected_provider, m),
            help="Select the model to use"
        )
        
        # Update model if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.info(f"Model changed to **{selected_model}**. Next message will use the new model.")
        
        # System messages toggle
        show_system = st.checkbox(
            "Show system messages",
            value=st.session_state.show_system_messages,
            help="Display system/debug messages in the chat"
        )
        if show_system != st.session_state.show_system_messages:
            st.session_state.show_system_messages = show_system
            st.rerun()

        st.divider()

        # 4. Chat History
        st.subheader("Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", help="Start a new conversation", use_container_width=True):
            # Create new thread
            new_thread_id = str(uuid4())
            
            # Create new chat in storage
            storage = get_chat_storage()
            new_chat = Chat(thread_id=new_thread_id)
            storage.upsert_chat(new_chat)
            
            # Update session state
            st.session_state.thread_id = new_thread_id
            st.session_state.messages = []
            st.session_state.welcome_initialized = False  # Reset welcome message flag
            st.rerun()
        
        # List saved chats
        storage = get_chat_storage()
        chats = storage.list_chats(limit=20)
        
        # Rename form (shown when editing_thread_id is set)
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
                        # Update chat title
                        if new_title.strip():
                            editing_chat.title = new_title.strip()
                        else:
                            editing_chat.title = "New Chat"
                        editing_chat.updated_at = datetime.now()
                        storage.upsert_chat(editing_chat)
                        # Clear editing state
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
                # Highlight current chat
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
                        # Switch to selected chat
                        st.session_state.thread_id = chat.thread_id
                        st.session_state.messages = storage.load_messages(chat.thread_id)
                        st.session_state.welcome_initialized = True  # Don't show welcome for loaded chats
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

       
        # 5. Information
        st.subheader("About")
        st.info(
            """
            **JüNA Chatbot** is a production-ready neutron science chatbot
            built with LangGraph, FastAPI, and Streamlit.
            
            It supports custom LangGraph agents to power
            tailored neutron science chatbot experiences.
            """
        )
