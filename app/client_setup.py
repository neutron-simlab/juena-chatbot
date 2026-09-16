"""Client construction for the JüNA Streamlit UI.

Core ships `juena_core.ui.client_setup.initialize_client`; this wrapper supplies
this application's subclass because every page also calls `/auth/me` and
`/research` (00, decision 10). The shared factory settles construction while
the application still supplies the three things core refuses to guess: client
class, agent id and timeout.

`JUENA_AGENT_ID` stays in this application deliberately: a shared package must
not smuggle one application's identity into another. `GET /agents` reports the
authoritative default, but it requires an authenticated session and the client
has to exist before the user signs in, so the id is named here. Retired ids
still resolve server side.
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
    """Initialize the JüNA client against the private API URL."""
    return initialize_core_client(
        internal_api_url,
        agent_id,
        session_token=session_token,
        timeout=float(global_config.TIMEOUT_SECONDS),
        client_class=AgentClient,
    )
