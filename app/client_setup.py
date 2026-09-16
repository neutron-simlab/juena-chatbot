"""Client construction for the JüNA Streamlit UI.

Core ships `juena_core.ui.client_setup.initialize_client`, and this application
cannot use it: it builds a `BaseAgentClient`, while every page here also calls
`/auth/me` and `/research`, which live on this application's subclass (00,
decision 10). What core's version *does* settle is the two arguments it refuses
to guess — the agent id and the timeout — and both of those answers are here.

`JUENA_AGENT_ID` stays in this application deliberately: a shared package must
not smuggle one application's identity into another. `GET /agents` reports the
authoritative default, but it requires an authenticated session and the client
has to exist before the user signs in, so the id is named here. Retired ids
still resolve server side.
"""

from __future__ import annotations

from juena.clients.client import AgentClient
from juena.core.config import global_config

__all__ = ["JUENA_AGENT_ID", "initialize_client"]

JUENA_AGENT_ID = "juena"


def initialize_client(
    internal_api_url: str,
    agent_id: str = JUENA_AGENT_ID,
    session_token: str | None = None,
) -> AgentClient:
    """Initialize the JüNA client against the private API URL."""
    return AgentClient(
        base_url=internal_api_url,
        agent=agent_id,
        session_token=session_token,
        timeout=float(global_config.TIMEOUT_SECONDS),
    )
