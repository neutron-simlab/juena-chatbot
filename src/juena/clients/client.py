import json
import weakref
from collections.abc import Generator, Sequence
from typing import Any

import httpx
from httpx_sse import EventSource, SSEError

from juena.schema.llm_models import Provider
from juena.schema.server import (
    ChatMessage,
    StreamInput,
)


class AgentClientError(Exception):
    pass


def _provider_value(provider: str | Provider | None) -> str | None:
    if provider is None:
        return None
    if isinstance(provider, Provider):
        return provider.value
    return provider


def _http_error_message(exc: httpx.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return f"Error: {detail}"
    return f"Error: {exc}"


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        session_token: str | None = None,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.agent: str | None = agent
        self.session_token = session_token
        self._client = httpx.Client()
        self._client_finalizer = weakref.finalize(self, self._client.close)

    def close(self) -> None:
        """Close the reusable HTTP connection pool."""
        if self._client_finalizer.alive:
            self._client_finalizer()

    def __enter__(self) -> "AgentClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def health(self) -> bool:
        """Return whether the API health endpoint is reachable."""
        try:
            response = self._client.get(
                f"{self.base_url}/health",
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.session_token:
            headers["Cookie"] = f"juena_session={self.session_token}"
        return headers

    def _build_attachment_payloads(
        self,
        attachments: Sequence[Any] | None,
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        payloads: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, attachment in enumerate(attachments or [], start=1):
            filename = str(getattr(attachment, "name", "") or f"attachment_{index}.txt")
            content_type = str(
                getattr(attachment, "type", None)
                or getattr(attachment, "content_type", None)
                or "text/plain"
            )
            if hasattr(attachment, "getvalue"):
                content = attachment.getvalue()
            elif hasattr(attachment, "read"):
                content = attachment.read()
                seek = getattr(attachment, "seek", None)
                if callable(seek):
                    seek(0)
            else:
                raise AgentClientError(f"Unsupported attachment object for '{filename}'")

            if not isinstance(content, bytes):
                raise AgentClientError(f"Attachment '{filename}' did not provide bytes")
            payloads.append(("attachments", (filename, content, content_type)))
        return payloads

    def _build_file_request_data(
        self,
        message: str,
        model: str | None,
        provider: str | Provider | None,
        thread_id: str | None,
    ) -> dict[str, str]:
        data = {"message": message}
        provider_value = _provider_value(provider)
        if thread_id:
            data["thread_id"] = thread_id
        if model:
            data["model"] = model
        if provider_value:
            data["provider"] = provider_value
        return data

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

    def list_chats(self, limit: int = 50) -> list[dict[str, Any]]:
        """List conversations owned by the authenticated user."""
        try:
            response = self._client.get(
                f"{self.base_url}/chats",
                params={"limit": limit},
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        payload = response.json()
        if not isinstance(payload, list):
            raise AgentClientError("Error: Invalid chat-list response from server")
        return payload

    def create_chat(self, thread_id: str, title: str = "New Chat") -> dict[str, Any]:
        """Create an authenticated, user-owned conversation."""
        try:
            response = self._client.post(
                f"{self.base_url}/chats",
                json={"thread_id": thread_id, "title": title},
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        return response.json()

    def get_chat(self, thread_id: str, include_messages: bool = True) -> dict[str, Any] | None:
        """Load an owned conversation, optionally without its message history."""
        try:
            response = self._client.get(
                f"{self.base_url}/chats/{thread_id}",
                params={"include_messages": include_messages},
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        return response.json()

    def update_chat(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Update metadata on an owned conversation."""
        payload = {
            key: value
            for key, value in {"title": title, "summary": summary}.items()
            if value is not None
        }
        try:
            response = self._client.patch(
                f"{self.base_url}/chats/{thread_id}",
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        return response.json()

    def _parse_sse_data(self, data: str) -> ChatMessage | dict | None:
        """
        Parse one SSE `data` payload into a message, token, or status dict.

        Returns None only for payloads that carry nothing to render; callers
        must skip those rather than treating them as end-of-stream.
        """
        try:
            parsed = json.loads(data)
        except ValueError as e:
            raise AgentClientError(f"Error JSON parsing message from server: {e}")

        parsed_type = parsed.get("type", "")

        if parsed_type == "message":
            try:
                return ChatMessage.model_validate(parsed["content"])
            except Exception as e:
                raise AgentClientError(f"Server returned invalid message: {e}")

        if parsed_type == "error":
            return ChatMessage(type="ai", content="Error: " + parsed.get("content", "Unknown error"))

        if parsed_type == "thread":
            return {"type": "thread", "thread_id": parsed.get("thread_id", "")}

        if parsed_type == "token":
            return {"type": "token", "content": parsed.get("content", "")}

        if parsed_type == "status":
            return {
                "type": "status",
                "phase": parsed.get("phase", ""),
                "label": parsed.get("label", ""),
                "tool": parsed.get("tool"),
                "summary": parsed.get("summary"),
            }

        if parsed_type == "sandbox_status":
            return {
                "type": "sandbox_status",
                "status": parsed.get("status", ""),
                "label": parsed.get("label", ""),
                "job_id": parsed.get("job_id"),
                "exit_code": parsed.get("exit_code"),
            }

        if parsed_type == "approval_required":
            return {
                "type": "approval_required",
                "interrupt_id": parsed.get("interrupt_id", ""),
                "action_name": parsed.get("action_name", ""),
                "command": parsed.get("command", ""),
                "description": parsed.get("description", ""),
                "limits": parsed.get("limits") or {},
                "allowed_decisions": parsed.get("allowed_decisions") or [],
            }

        if parsed_type == "artifact":
            return {
                "type": "artifact",
                "artifact_id": parsed.get("artifact_id", ""),
                "filename": parsed.get("filename", "plot.png"),
                "mime_type": parsed.get("mime_type", "image/png"),
                "kind": parsed.get("kind", "image"),
                "size": parsed.get("size"),
                "width": parsed.get("width"),
                "height": parsed.get("height"),
                "caption": parsed.get("caption", "Plot"),
                "created_at": parsed.get("created_at"),
                "content_base64": parsed.get("content_base64", ""),
            }

        return {"type": "unknown", "content": parsed.get("content", str(parsed))}

    def _build_stream_request(
        self,
        message: str,
        model: str | None,
        provider: str | None,
        thread_id: str | None,
        attachments: Sequence[Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build the URL and httpx kwargs shared by `stream` and `astream`."""
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        stream_kwargs: dict[str, Any] = {
            "headers": self._headers,
            # A streamed response may legitimately stay open while the agent
            # works, but connection setup and writes must still be bounded.
            "timeout": (
                httpx.Timeout(
                    connect=self.timeout,
                    read=None,
                    write=self.timeout,
                    pool=self.timeout,
                )
                if self.timeout is not None
                else None
            ),
        }

        if attachments:
            stream_kwargs["data"] = self._build_file_request_data(
                message, model, provider, thread_id
            )
            stream_kwargs["files"] = self._build_attachment_payloads(attachments)
            return f"{self.base_url}/{self.agent}/stream_with_files", stream_kwargs

        request = StreamInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if provider:
            request.provider = Provider(provider) if isinstance(provider, str) else provider
        stream_kwargs["json"] = request.model_dump(exclude_none=True)
        return f"{self.base_url}/{self.agent}/stream", stream_kwargs

    def stream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> Generator[ChatMessage | dict, None, None]:
        """
        Stream the agent's response synchronously.

        Yields complete messages as `ChatMessage`, and `token`/`status`/`thread`
        dicts as the run progresses.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            provider (str, optional): LLM provider to use (openai or blablador)
            thread_id (str, optional): Thread ID for continuing a conversation
            attachments (Sequence, optional): Files to stage for this turn

        Returns:
            Generator[ChatMessage | dict, None, None]: The response from the agent
        """
        url, stream_kwargs = self._build_stream_request(
            message, model, provider, thread_id, attachments
        )
        try:
            with self._client.stream("POST", url, **stream_kwargs) as response:
                response.raise_for_status()
                for sse in EventSource(response).iter_sse():
                    parsed = self._parse_sse_data(sse.data)
                    if parsed is not None:
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        except SSEError as e:
            raise AgentClientError(f"Error: {e}")

    def resume_stream(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        decision: str,
        edited_command: str | None = None,
        model: str | None = None,
        provider: str | Provider | None = None,
    ) -> Generator[ChatMessage | dict, None, None]:
        """Resume a pending sandbox command after a user decision."""
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")

        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "interrupt_id": interrupt_id,
            "decision": decision,
        }
        if edited_command is not None:
            payload["edited_command"] = edited_command
        if model:
            payload["model"] = model
        if provider_value := _provider_value(provider):
            payload["provider"] = provider_value

        timeout = (
            httpx.Timeout(connect=self.timeout, read=None, write=self.timeout, pool=self.timeout)
            if self.timeout is not None
            else None
        )
        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/{self.agent}/resume",
                json=payload,
                headers=self._headers,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                for sse in EventSource(response).iter_sse():
                    parsed = self._parse_sse_data(sse.data)
                    if parsed is not None:
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        except SSEError as e:
            raise AgentClientError(f"Error: {e}")

    def get_artifact(self, artifact_id: str) -> bytes:
        """Download one generated image owned by the current user."""
        try:
            response = self._client.get(
                f"{self.base_url}/artifacts/{artifact_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))
        if not response.content:
            raise AgentClientError("Error: Server returned an empty generated file")
        return response.content

    def get_pending_approval(
        self,
        thread_id: str,
        *,
        model: str | None = None,
        provider: str | Provider | None = None,
    ) -> dict[str, Any] | None:
        """Return a thread's pending sandbox confirmation, if one exists."""
        params = {
            key: value
            for key, value in {
                "model": model,
                "provider": _provider_value(provider),
            }.items()
            if value
        }
        try:
            response = self._client.get(
                f"{self.base_url}/threads/{thread_id}/pending-approval",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as e:
            if isinstance(e, httpx.HTTPError):
                raise AgentClientError(_http_error_message(e))
            raise AgentClientError("Error: Server returned an invalid approval response") from e
        approval = payload.get("approval") if isinstance(payload, dict) else None
        return approval if isinstance(approval, dict) else None

    def is_token_message(self, message: ChatMessage | str | dict) -> bool:
        """
        Check if a message is a token message.
        
        Args:
            message: Message to check
            
        Returns:
            True if message is a token, False otherwise
        """
        if isinstance(message, str):
            return True  # Legacy string token
        if isinstance(message, dict):
            msg_type = message.get("type", "")
            return msg_type == "token"
        return False

    def get_token_content(self, message: ChatMessage | str | dict) -> str | None:
        """
        Get the content from a token message.
        
        Args:
            message: Token message
            
        Returns:
            Token content string or None
        """
        if isinstance(message, str):
            return message  # Legacy string token
        if isinstance(message, dict) and self.is_token_message(message):
            return message.get("content")
        return None

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        """Delete all persisted server-side state for a conversation thread."""

        try:
            response = self._client.delete(
                f"{self.base_url}/threads/{thread_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))

        return response.json()
