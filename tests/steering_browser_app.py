"""Browser acceptance with the real JueNA chat page and an in-memory API."""
import os
from uuid import uuid4
import streamlit as st

os.environ["JUENA_SECRET_KEY"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
os.environ["SANDBOX_ENABLED"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGFUSE_TRACING"] = "false"

from juena.core.config import global_config
from juena_core.clients.base import BaseAgentClient
from app.chat_interface import render_chat_interface
from steering_fixture_api import start_fixture_api

fixture_api = st.cache_resource(start_fixture_api)


fixture_api()
st.session_state.setdefault("client", BaseAgentClient("http://127.0.0.1:19622", agent="juena", timeout=5))
st.session_state.setdefault("thread_id", str(uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("user", {"chat_ready":True})
st.session_state.setdefault("show_system_messages", False)
st.session_state.setdefault("selected_provider", "blablador")
st.session_state.setdefault("selected_model", global_config.DEFAULT_MODEL)
st.session_state.run_controls = True
render_chat_interface()
