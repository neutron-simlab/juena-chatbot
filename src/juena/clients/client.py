"""The JüNA HTTP client: core's transport, plus this application's routes.

`BaseAgentClient` covers every route juena-core serves: streaming, chats,
artifacts and interrupts. `/auth/me`, the personal credential routes and
`/research` are JüNA's own, so their methods live in this subclass.
"""

from __future__ import annotations

from typing import Any

import httpx
from urllib.parse import urlsplit

from juena.core.config import global_config

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

    def _credential_request(self, method: str, service: str, **kwargs: Any) -> dict:
        origin = urlsplit(global_config.UI_PUBLIC_URL)
        try:
            response = self._client.request(
                method, f"{self.base_url}/auth/me/credentials/{service}",
                headers={**self._headers, "Origin": f"{origin.scheme}://{origin.netloc}"},
                timeout=max(self.timeout, 35), **kwargs,
            )
        except httpx.HTTPError:
            raise AgentClientError("The service could not be reached. Try again shortly.") from None
        if response.is_error:
            try:
                detail = response.json().get("detail")
                message = detail.get("message") if isinstance(detail, dict) else detail
            except ValueError:
                message = None
            error = AgentAuthenticationError if response.status_code == 401 else AgentClientError
            raise error(message if isinstance(message, str) else "The credential change was refused.")
        return response.json()

    def save_api_key(self, service: str, key: str, endpoint: str) -> dict:
        """Have the server check the user's personal key for a service and save it.

        Args:
            service: Service name, e.g. ``"blablador"``.
            key: The API key.
            endpoint: The service address the profile page showed; the server
                refuses one that no longer matches its own.

        Returns:
            The user's key status for every service.

        Raises:
            AgentAuthenticationError: The session has expired.
            AgentClientError: The key was refused or the server was unreachable.
        """
        return self._credential_request("PUT", service, json={"key": key, "endpoint": endpoint})

    def remove_api_key(self, service: str, endpoint: str) -> dict:
        """Delete the user's saved key for a service.

        Args:
            service: Service name, e.g. ``"blablador"``.
            endpoint: The service address the profile page showed.

        Returns:
            The user's key status for every service.

        Raises:
            AgentAuthenticationError: The session has expired.
            AgentClientError: The change was refused or the server was unreachable.
        """
        return self._credential_request("DELETE", service, params={"endpoint": endpoint})

    def get_current_user(self) -> dict[str, Any]:
        """Return the user associated with the current session cookie.

        Returns:
            The user's id, email, display name, key status per service, and
            whether they can chat.

        Raises:
            AgentClientError: The request failed, including an expired session.
        """
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

        Polling failures are raised so the sidebar cannot mistake an expired
        session or unavailable server for an authoritative "nothing
        outstanding" result.

        Args:
            thread_id: The conversation to ask about.

        Returns:
            ``{"jobs": [...], "running": int, "ready": int}``.

        Raises:
            AgentAuthenticationError: The session has expired.
            AgentClientError: The server was unreachable or answered badly.
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
