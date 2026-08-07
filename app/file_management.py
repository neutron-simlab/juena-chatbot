"""Client initialization for the Streamlit UI."""

from juena.clients.client import AgentClient
from juena.core.config import global_config
from juena.server.agent_registry import DEFAULT_AGENT


def initialize_client(
    internal_api_url: str,
    agent_id: str = DEFAULT_AGENT,
    session_token: str | None = None,
) -> AgentClient:
    """Initialize AgentClient with the private API URL."""
    return AgentClient(
        base_url=internal_api_url,
        agent=agent_id,
        session_token=session_token,
        timeout=float(global_config.TIMEOUT_SECONDS),
    )
