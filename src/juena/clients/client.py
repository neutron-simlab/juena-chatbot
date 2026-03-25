import json
import os
from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any

import httpx

from juena.schema.llm_models import Provider
from juena.schema.server import (
    ChatMessage,
    StreamInput,
    UserInput,
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
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.agent: str | None = agent

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
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
        user_id: str | None,
    ) -> dict[str, str]:
        data = {"message": message}
        provider_value = _provider_value(provider)
        if thread_id:
            data["thread_id"] = thread_id
        if user_id:
            data["user_id"] = user_id
        if model:
            data["model"] = model
        if provider_value:
            data["provider"] = provider_value
        return data

    def update_agent(self, agent: str) -> None:
        """Update the agent to use for requests."""
        self.agent = agent

    def list_repositories(self) -> list[dict[str, Any]]:
        """Return configured repository metadata from the server."""
        try:
            response = httpx.get(
                f"{self.base_url}/repositories",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        try:
            payload = response.json()
        except ValueError as e:
            raise AgentClientError(f"Error: {e}")

        if not isinstance(payload, list):
            raise AgentClientError("Error: Invalid repository response from server")

        return payload

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            provider (str, optional): LLM provider to use (openai or blablador)
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        async with httpx.AsyncClient() as client:
            try:
                if attachments:
                    response = await client.post(
                        f"{self.base_url}/{self.agent}/invoke_with_files",
                        data=self._build_file_request_data(
                            message,
                            model,
                            provider,
                            thread_id,
                            user_id,
                        ),
                        files=self._build_attachment_payloads(attachments),
                        headers=self._headers,
                        timeout=self.timeout,
                    )
                else:
                    request = UserInput(message=message)
                    if thread_id:
                        request.thread_id = thread_id
                    if model:
                        request.model = model  # type: ignore[assignment]
                    if provider:
                        if isinstance(provider, str):
                            request.provider = Provider(provider)
                        else:
                            request.provider = provider
                    if agent_config:
                        request.agent_config = agent_config
                    if user_id:
                        request.user_id = user_id
                    response = await client.post(
                        f"{self.base_url}/{self.agent}/invoke",
                        json=request.model_dump(),
                        headers=self._headers,
                        timeout=self.timeout,
                    )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(_http_error_message(e))

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            provider (str, optional): LLM provider to use (openai or blablador)
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        try:
            if attachments:
                response = httpx.post(
                    f"{self.base_url}/{self.agent}/invoke_with_files",
                    data=self._build_file_request_data(
                        message,
                        model,
                        provider,
                        thread_id,
                        user_id,
                    ),
                    files=self._build_attachment_payloads(attachments),
                    headers=self._headers,
                    timeout=self.timeout,
                )
            else:
                request = UserInput(message=message)
                if thread_id:
                    request.thread_id = thread_id
                if model:
                    request.model = model  # type: ignore[assignment]
                if provider:
                    if isinstance(provider, str):
                        request.provider = Provider(provider)
                    else:
                        request.provider = provider
                if agent_config:
                    request.agent_config = agent_config
                if user_id:
                    request.user_id = user_id
                response = httpx.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | dict | None:
        """
        Parse a single SSE line into a normalized message or token.
        
        Returns:
            ChatMessage, normalized token dict, or None
        """
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            
            parsed_type = parsed.get("type", "")
            
            # Handle complete messages
            if parsed_type == "message":
                try:
                    return ChatMessage.model_validate(parsed["content"])
                except Exception as e:
                    raise Exception(f"Server returned invalid message: {e}")
            
            # Handle errors
            elif parsed_type == "error":
                error_msg = "Error: " + parsed.get("content", "Unknown error")
                return ChatMessage(type="ai", content=error_msg)
            
            # Thread ID for conversation persistence (sent first in stream; use for follow-up)
            elif parsed_type == "thread":
                return {"type": "thread", "thread_id": parsed.get("thread_id", "")}
            
            # Token streaming
            elif parsed_type == "token" or parsed_type.startswith("token"):
                content = parsed.get("content", "")
                return {"type": "token", "content": content}
            
            # Fallback for unknown types
            else:
                return {"type": "unknown", "content": parsed.get("content", str(parsed))}
        
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: Sequence[Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str | dict, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            provider (str, optional): LLM provider to use (openai or blablador)
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str | dict, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        try:
            stream_kwargs: dict[str, Any] = {
                "headers": self._headers,
                "timeout": self.timeout,
            }
            if attachments:
                stream_kwargs["data"] = self._build_file_request_data(
                    message,
                    model,
                    provider,
                    thread_id,
                    user_id,
                )
                stream_kwargs["files"] = self._build_attachment_payloads(attachments)
                url = f"{self.base_url}/{self.agent}/stream_with_files"
            else:
                request = StreamInput(message=message, stream_tokens=stream_tokens)
                if thread_id:
                    request.thread_id = thread_id
                if user_id:
                    request.user_id = user_id
                if model:
                    request.model = model  # type: ignore[assignment]
                if provider:
                    if isinstance(provider, str):
                        request.provider = Provider(provider)
                    else:
                        request.provider = provider
                if agent_config:
                    request.agent_config = agent_config
                stream_kwargs["json"] = request.model_dump()
                url = f"{self.base_url}/{self.agent}/stream"

            with httpx.stream("POST", url, **stream_kwargs) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(_http_error_message(e))

    async def astream(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: Sequence[Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str | dict, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            provider (str, optional): LLM provider to use (openai or blablador)
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str | dict, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        async with httpx.AsyncClient() as client:
            try:
                stream_kwargs: dict[str, Any] = {
                    "headers": self._headers,
                    "timeout": self.timeout,
                }
                if attachments:
                    stream_kwargs["data"] = self._build_file_request_data(
                        message,
                        model,
                        provider,
                        thread_id,
                        user_id,
                    )
                    stream_kwargs["files"] = self._build_attachment_payloads(attachments)
                    url = f"{self.base_url}/{self.agent}/stream_with_files"
                else:
                    request = StreamInput(message=message, stream_tokens=stream_tokens)
                    if thread_id:
                        request.thread_id = thread_id
                    if model:
                        request.model = model  # type: ignore[assignment]
                    if provider:
                        if isinstance(provider, str):
                            request.provider = Provider(provider)
                        else:
                            request.provider = provider
                    if agent_config:
                        request.agent_config = agent_config
                    if user_id:
                        request.user_id = user_id
                    stream_kwargs["json"] = request.model_dump()
                    url = f"{self.base_url}/{self.agent}/stream"
                async with client.stream("POST", url, **stream_kwargs) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(_http_error_message(e))

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
    
    def restart(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """
        Restart the agent with new provider/model configuration.
        
        This method forces reinitialization of the agent graph with new LLM configuration,
        similar to refreshing the web page but keeping the new provider/model.
        
        Args:
            model (str, optional): New LLM model name to use
            provider (str, optional): New LLM provider to use (openai or blablador)
        
        Returns:
            dict: Dictionary with restart status and agent info
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        
        # Build query parameters
        params: dict[str, str] = {}
        if model:
            params["model"] = model
        if provider:
            params["provider"] = provider
        
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/restart",
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error restarting agent: {e}")
        
        return response.json()
