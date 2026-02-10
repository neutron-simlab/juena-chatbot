"""
File management functions for Streamlit UI.

This module provides functions for server health checks and client initialization.
"""
import streamlit as st
import httpx
from typing import Optional

from juena.clients.client import AgentClient
from juena.server.agent_registry import DEFAULT_AGENT

def check_server_health(server_url: str) -> bool:
    """Check if server is running by hitting /health endpoint."""
    try:
        response = httpx.get(f"{server_url}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def initialize_client(server_url: str, agent_id: str = DEFAULT_AGENT) -> Optional[AgentClient]:
    """Initialize AgentClient with server URL."""
    try:
        return AgentClient(base_url=server_url, agent=agent_id)
    except Exception as e:
        st.error(f"Failed to initialize client: {e}")
        return None
