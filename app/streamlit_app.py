"""
Streamlit UI for JüNA Chatbot

This Streamlit application provides a web interface for interacting with
LangGraph agents through the FastAPI service.
"""
import streamlit as st
from uuid import uuid4
from pathlib import Path
import sys

# Add parent directory to path to import juena modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from juena.schema.llm_models import Provider
from juena.core.config import global_config
from juena.core.llms_providers import get_available_providers, get_default_model

# Import UI modules
from app.sidebar import render_sidebar
from app.chat_interface import render_chat_interface
from app.file_management import check_server_health, initialize_client
from app.chat_storage import get_chat_storage, Chat

# Paths and assets
_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None

# Page configuration
st.set_page_config(
    page_title="JüNA Chatbot",
    page_icon=str(_logo_path) if _logo_path and _logo_path.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())

if "server_url" not in st.session_state:
    st.session_state.server_url = "http://localhost:8000"

if "client" not in st.session_state:
    st.session_state.client = None

if "server_connected" not in st.session_state:
    st.session_state.server_connected = False

if "show_system_messages" not in st.session_state:
    st.session_state.show_system_messages = False

if "selected_provider" not in st.session_state:
    # Initialize provider from available providers
    available_providers = get_available_providers()
    available_provider_list = [p.value for p in Provider if available_providers.get(p.value, False)]
    
    # Use global_config.DEFAULT_PROVIDER if available, otherwise use first available
    if global_config.DEFAULT_PROVIDER.lower() in available_provider_list:
        st.session_state.selected_provider = global_config.DEFAULT_PROVIDER.lower()
    elif available_provider_list:
        st.session_state.selected_provider = available_provider_list[0]
    else:
        # Fallback to OpenAI if no providers available (will show error in sidebar)
        st.session_state.selected_provider = Provider.OPENAI.value

if "selected_model" not in st.session_state:
    # Initialize model based on selected provider using registry
    st.session_state.selected_model = get_default_model(st.session_state.selected_provider) or "gpt-4o-mini"

# Initialize chat storage
if "chat_storage" not in st.session_state:
    st.session_state.chat_storage = get_chat_storage()

# Create or load current chat on first run
if "chat_initialized" not in st.session_state:
    storage = st.session_state.chat_storage
    
    # Create a new chat entry for the current thread
    existing_chat = storage.get_chat(st.session_state.thread_id)
    if existing_chat is None:
        new_chat = Chat(thread_id=st.session_state.thread_id)
        storage.upsert_chat(new_chat)
    else:
        # Load existing messages if resuming a chat
        st.session_state.messages = storage.load_messages(st.session_state.thread_id)
    
    st.session_state.chat_initialized = True

# Auto-connect to server on app load (only check once per session)
if not hasattr(st.session_state, '_health_checked'):
    st.session_state.server_connected = check_server_health(st.session_state.server_url)
    if st.session_state.server_connected:
        st.session_state.client = initialize_client(st.session_state.server_url)
        if st.session_state.client is None:
            st.session_state.server_connected = False
    else:
        st.session_state.client = None
    st.session_state._health_checked = True

# Render sidebar
render_sidebar()

# Render main chat interface
render_chat_interface()
