"""Client construction for the JüNA Streamlit UI.

Wraps `juena_core.ui.client_setup.initialize_client` with the three things core
does not guess: the client class (this application's subclass, because every
page also calls `/auth/me` and `/research`), the agent id and the timeout.

The agent id is named here because the client is built before the user signs
in, and `GET /agents`, which reports the default, needs a signed-in session.
"""

from __future__ import annotations

from juena.clients.client import AgentClient
from juena.core.config import global_config
from juena_core.ui.client_setup import initialize_client as initialize_core_client

__all__ = ["JUENA_AGENT_ID", "initialize_client"]

JUENA_AGENT_ID = "juena"


def initialize_client(
    internal_api_url: str,
    agent_id: str = JUENA_AGENT_ID,
    session_token: str | None = None,
) -> AgentClient:
    """Initialize the JüNA client against the private API URL.

    Args:
        internal_api_url: Server-side URL of the API, not the browser-facing one.
        agent_id: Agent the client talks to.
        session_token: The user's session cookie; None before sign-in.

    Returns:
        The client, using this application's timeout.
    """
    return initialize_core_client(
        internal_api_url,
        agent_id,
        session_token=session_token,
        timeout=float(global_config.TIMEOUT_SECONDS),
        client_class=AgentClient,
    )
