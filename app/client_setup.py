"""Client initialization for the Streamlit UI."""

from juena.clients.client import AgentClient
from juena.core.config import global_config

# The UI talks to the API over HTTP and must not import the server package to
# learn this: the two run as separate processes, and in production as separate
# containers. `GET /agents` reports the authoritative default, but it requires
# an authenticated session and the client has to exist before the user signs
# in, so the id is named here. Retired ids still resolve server side.
JUENA_AGENT_ID = "juena"


def initialize_client(
    internal_api_url: str,
    agent_id: str = JUENA_AGENT_ID,
    session_token: str | None = None,
) -> AgentClient:
    """Initialize AgentClient with the private API URL."""
    return AgentClient(
        base_url=internal_api_url,
        agent=agent_id,
        session_token=session_token,
        timeout=float(global_config.TIMEOUT_SECONDS),
    )
