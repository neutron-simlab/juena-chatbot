"""The JüNA HTTP client: core's transport, plus this application's two routes.

`BaseAgentClient` owns everything behind a route juena-core serves — the
streaming transport, the SSE vocabulary, chats, artifacts and interrupts. It
deliberately stops there (00-BOUNDARY.md, decision 10): `/auth/me` and
`/research` are JüNA's, and a client method with no route behind it is a 404
nobody notices until a user clicks it.

So this subclass adds exactly those two, and nothing else.
"""

from __future__ import annotations

from typing import Any

import httpx

from juena_core.clients.base import (
    AgentAuthenticationError,
    AgentClientError,
    BaseAgentClient,
    _http_error_message,
)

__all__ = [
    "AgentClient",
    "AgentClientError",
    "AgentAuthenticationError",
]


class AgentClient(BaseAgentClient):
    """Core's client plus JüNA's authentication and background-research routes."""

    def get_current_user(self) -> dict[str, Any]:
        """Return the user associated with the current session cookie."""
        try:
            response = self._client.get(
                f"{self.base_url}/auth/me",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        return response.json()

    def list_research(self, thread_id: str) -> dict[str, Any]:
        """Background research on a thread that has not been reported yet.

        Returns ``{"jobs": [...], "running": int, "ready": int}``. Polling
        failures are raised so the sidebar cannot mistake an expired session or
        unavailable server for an authoritative "nothing outstanding" result.
        """
        try:
            response = self._client.get(
                f"{self.base_url}/research",
                params={"thread_id": thread_id},
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise AgentAuthenticationError(_http_error_message(exc)) from exc
            raise AgentClientError(_http_error_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise AgentClientError(_http_error_message(exc)) from exc
        except ValueError as exc:
            raise AgentClientError(
                "Error: Server returned invalid research status"
            ) from exc
        if not isinstance(payload, dict):
            raise AgentClientError("Error: Server returned invalid research status")
        return payload
