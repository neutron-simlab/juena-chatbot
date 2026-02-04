"""
Sidebar configuration UI for Streamlit app.

This module provides the sidebar UI for server configuration, LLM settings,
and thread management.
"""
import streamlit as st
from uuid import uuid4
from pathlib import Path

from juena.schema.llm_models import (
    Provider,
    OpenAIModelName,
    BlabladorModelName,
    get_blablador_model_display_name,
    get_default_model_for_provider,
)
from juena.core.llms_providers import get_available_providers, get_available_models
from app.file_management import (
    check_server_health,
    initialize_client,
)

# Paths and assets
_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None


def render_sidebar() -> None:
    """Render the sidebar with all configuration options."""
    with st.sidebar:
        if _logo_path and _logo_path.exists():
            st.image(str(_logo_path), width='stretch')
        st.title("JueNA")
        
        # Connection status indicator
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
        else:
            st.error("🔴 Server Disconnected")
        
        st.divider()

        # 1. Server Configuration
        st.subheader("Server Configuration")
        server_url = st.text_input(
            "Server URL",
            value=st.session_state.server_url,
            help="URL of the chatbot API server",
            key="server_url_input"
        )
        
        if server_url != st.session_state.server_url:
            st.session_state.server_url = server_url
            # Re-check health and reinitialize client
            st.session_state.server_connected = check_server_health(server_url)
            if st.session_state.server_connected:
                st.session_state.client = initialize_client(server_url)
                if st.session_state.client is None:
                    st.session_state.server_connected = False
            else:
                st.session_state.client = None
            st.rerun()
        
        if st.button("Reconnect", help="Reconnect to the server"):
            st.session_state.server_connected = check_server_health(st.session_state.server_url)
            if st.session_state.server_connected:
                st.session_state.client = initialize_client(st.session_state.server_url)
                if st.session_state.client is None:
                    st.session_state.server_connected = False
            else:
                st.session_state.client = None
            st.rerun()

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

        # 3. Thread Management
        st.subheader("Thread Management")
        st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
        st.text(f"User ID: {st.session_state.user_id[:8]}...")
        
        if st.button("New Thread", help="Start a new conversation thread"):
            st.session_state.thread_id = str(uuid4())
            st.session_state.messages = []
            st.session_state.current_interrupt = None
            st.rerun()

        st.divider()

        # 4. Settings
        st.subheader("Settings")
        show_system = st.checkbox(
            "Show System Messages",
            value=st.session_state.show_system_messages,
            help="Display system messages in chat history (for debugging)"
        )
        if show_system != st.session_state.show_system_messages:
            st.session_state.show_system_messages = show_system

        st.divider()

        # 5. Information
        st.subheader("About")
        st.info(
            """
            **Chatbot Template** is a general-purpose chatbot framework
            built with LangGraph, FastAPI, and Streamlit.
            
            You can plug in your own LangGraph agents to create
            custom chatbot applications.
            """
        )
