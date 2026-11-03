"""
Streamlit UI for JüNA Chatbot

This Streamlit application provides a web interface for interacting with
LangGraph agents through the FastAPI service.
"""
import streamlit as st
from uuid import uuid4
from pathlib import Path
import sys
from urllib.parse import urlencode

# Add parent directory to path to import juena modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from juena.core.config import global_config
from juena_core.llms_providers import get_default_model

from app.sidebar import render_sidebar
from app.profile_page import render_profile_page, render_blablador_guidance
from app.chat_interface import render_chat_interface
from app.client_setup import initialize_client
from app.ui_components import FILES_PANEL_KEY, render_files_panel_styles
from juena_core.ui.chat_storage import get_chat_storage
from juena_core.ui.filesystem import render_filesystem

_assets_dir = Path(__file__).parent / "assets"
_logo_path = _assets_dir / "logo.png" if (_assets_dir.exists() and (_assets_dir / "logo.png").exists()) else None

st.set_page_config(
    page_title="JüNA Chatbot",
    page_icon=str(_logo_path) if _logo_path and _logo_path.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Establish the private backend client and authenticate before initializing chat state.
session_token = st.context.cookies.get("juena_session")
internal_api_url = global_config.JUENA_API_INTERNAL_URL.rstrip("/")
client = st.session_state.get("client")
if (
    client is None
    or client.base_url != internal_api_url
    or client.session_token != session_token
):
    if client is not None:
        client.close()
    client = initialize_client(
        internal_api_url,
        session_token=session_token,
    )
    st.session_state.client = client
    st.session_state.filesystem_session_scope = str(uuid4())

if not client.health():
    st.error("The JüNA service is unavailable. Please try again later.")
    st.stop()

_login_route = "/auth/dev-login" if global_config.AUTH_DEV_BYPASS else "/auth/login"
_login_query = urlencode({"return_to": global_config.UI_PUBLIC_URL})
login_url = f"{global_config.JUENA_PUBLIC_URL.rstrip('/')}{_login_route}?{_login_query}"
if not session_token:
    render_blablador_guidance()
    if global_config.AUTH_DEV_BYPASS:
        st.warning(
            "Development mode: authentication is bypassed. Anyone reaching this "
            "page is signed in as the local developer account."
        )
        st.link_button("Sign in (development)", login_url, type="primary")
    else:
        st.info("Sign in with iffLogin to use JüNA.")
        st.link_button("Sign in with iffLogin", login_url, type="primary")
    st.stop()

try:
    current_user = client.get_current_user()
except Exception:
    st.warning("Your JüNA session is missing or has expired.")
    st.link_button("Sign in again", login_url, type="primary")
    st.stop()

st.session_state.session_token = session_token
st.session_state.user = current_user
st.session_state.setdefault("filesystem_session_scope", str(uuid4()))

if "messages" not in st.session_state:
    st.session_state.messages = []

# The active thread lives in the URL, not only in st.session_state: session
# state is per-Streamlit-session, so a browser reload would otherwise start a
# brand new conversation every time.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.query_params.get("thread") or str(uuid4())

if "show_system_messages" not in st.session_state:
    st.session_state.show_system_messages = False

st.session_state.selected_provider = "blablador"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = get_default_model("blablador")

# Initialize the remote, user-owned chat storage adapter.
st.session_state.chat_storage = get_chat_storage(client)

# Resume the thread named in the URL, if it exists and is ours. A thread that
# has no row yet is simply an unsent conversation: the row is created server
# side by the first invocation, so reloading an empty chat never leaves a
# stray "Chat f84f216a" behind in the sidebar.
if "chat_initialized" not in st.session_state:
    loaded = st.session_state.chat_storage.load_chat_with_messages(
        st.session_state.thread_id
    )
    if loaded is not None:
        _, st.session_state.messages = loaded
        st.session_state.welcome_initialized = True
    else:
        # A stale bookmark may refer to a deleted conversation identity.
        st.session_state.thread_id = str(uuid4())
    st.session_state.chat_initialized = True

# Keep the address bar in step with the active conversation so reloads and
# bookmarks return to it.
if st.query_params.get("thread") != st.session_state.thread_id:
    st.query_params["thread"] = st.session_state.thread_id

render_sidebar()
if st.session_state.get("current_page") == "profile":
    render_profile_page()
    st.stop()
render_files_panel_styles()
chat_column, files_column = st.columns([3, 2], gap="medium")
# Mount the independent browser first: rendering chat can consume SSE for minutes.
with files_column, st.container(key=FILES_PANEL_KEY):
    render_filesystem(
        api_url=global_config.JUENA_PUBLIC_URL,
        thread_id=st.session_state.thread_id,
        session_scope=st.session_state.filesystem_session_scope,
        refresh_generation=st.session_state.get("filesystem_refresh", {}).get(st.session_state.thread_id, 0),
    )
with chat_column:
    render_chat_interface()
