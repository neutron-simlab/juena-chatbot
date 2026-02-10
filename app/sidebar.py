"""
Sidebar configuration UI for Streamlit app.
"""
import streamlit as st
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from juena.schema.llm_models import (
    Provider,
    OpenAIModelName,
    BlabladorModelName,
    get_blablador_model_display_name,
    get_default_model_for_provider,
)
from juena.core.llms_providers import get_available_providers, get_available_models
from app.chat_storage import get_chat_storage, Chat

# Paths and assets
_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None


def render_sidebar() -> None:
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        if _logo_path and _logo_path.exists():
            st.image(str(_logo_path), width='stretch')
        st.title("JüNA")
        
        # 1 Connection status indicator
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
        else:
            st.error("🔴 Server Disconnected")
        
        st.divider()

        # 2. LLM Configuration
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
            try:
                provider_enum = Provider(st.session_state.selected_provider)
                st.session_state.selected_model = get_default_model_for_provider(provider_enum)
            except ValueError:
                pass
            st.warning(f"⚠️ Previously selected provider is unavailable. Auto-selected: **{st.session_state.selected_provider}**")
        
        selected_provider = st.radio(
            "Provider",
            options=provider_options,
            index=provider_options.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_options else 0,
            help="Select the LLM provider to use"
        )
        
        # Handle provider change
        if selected_provider != st.session_state.selected_provider:
            st.session_state.selected_provider = selected_provider
            if selected_provider == Provider.BLABLADOR.value:
                st.session_state.selected_model = BlabladorModelName.GPT_OSS.value
            else:
                st.session_state.selected_model = OpenAIModelName.GPT_4O_MINI.value
            st.info(f"Switched to **{selected_provider}**. Next message will use the new model.")
        
        # Model selector based on provider
        model_options = get_available_models(st.session_state.selected_provider)
        
        if st.session_state.selected_provider == Provider.OPENAI.value:
            # Ensure selected model is valid for current provider
            if st.session_state.selected_model not in model_options:
                default_model = get_default_model_for_provider(Provider.OPENAI)
                st.session_state.selected_model = default_model if default_model in model_options else (model_options[0] if model_options else OpenAIModelName.GPT_4O_MINI.value)
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
                help="Select the OpenAI model to use"
            )
        else:  # Blablador
            if st.session_state.selected_model not in model_options:
                default_model = get_default_model_for_provider(Provider.BLABLADOR)
                st.session_state.selected_model = default_model if default_model in model_options else (model_options[0] if model_options else BlabladorModelName.GPT_OSS.value)
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
                format_func=get_blablador_model_display_name,
                help="Select the Blablador model to use",
            )
        
        # Update model if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.info(f"Model changed to **{selected_model}**. Next message will use the new model.")

        st.divider()

        # 3. Chat History
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
                
                # Format display text
                display_title = chat.title if chat.title != "New Chat" else f"Chat {chat.thread_id[:8]}..."
                if len(display_title) > 25:
                    display_title = display_title[:22] + "..."
                
                # Layout: [select button] [write button] [delete button]
                select_col, write_col, delete_col = st.columns([5, 1, 1])
                
                with select_col:
                    button_type = "primary" if is_current else "secondary"
                    if st.button(
                        f"{'💬 ' if is_current else ''}{display_title}",
                        key=f"chat_{chat.thread_id}",
                        use_container_width=True,
                        type=button_type,
                        disabled=is_current
                    ):
                        # Switch to selected chat
                        st.session_state.thread_id = chat.thread_id
                        st.session_state.messages = storage.load_messages(chat.thread_id)
                        st.session_state.welcome_initialized = True  # Don't show welcome for loaded chats
                        st.rerun()
                
                with write_col:
                    if st.button(
                        "✏️",
                        key=f"write_{chat.thread_id}",
                        help="Rename this conversation"
                    ):
                        st.session_state.editing_thread_id = chat.thread_id
                        st.rerun()
                
                with delete_col:
                    if st.button(
                        "🗑",
                        key=f"delete_{chat.thread_id}",
                        help="Delete this conversation"
                    ):
                        # Delete chat from storage
                        storage.delete_chat(chat.thread_id)
                        
                        # If deleting the current chat, reset to a fresh thread
                        if is_current:
                            st.session_state.thread_id = str(uuid4())
                            st.session_state.messages = []
                            st.session_state.welcome_initialized = False
                        st.rerun()
        else:
            st.caption("No saved conversations yet.")
        
        st.divider()

       
        # 4. Information
        st.subheader("About")
        st.info(
            """
            **JüNA Chatbot Template** is a general-purpose chatbot framework
            built with LangGraph, FastAPI, and Streamlit.
            
            You can plug in your own LangGraph agents to create
            custom chatbot applications.
            """
        )
