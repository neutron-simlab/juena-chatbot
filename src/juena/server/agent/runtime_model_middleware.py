"""
Runtime model selection middleware for LangChain/LangGraph agents.

The UI owns provider/model selection. The server passes that selection through
LangChain runtime context on every invocation, and this middleware swaps the
request model at call time so compiled agent graphs do not need to be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from juena.core.llms_providers import build_chat_model
from juena.core.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeModelContext:
    """Per-request runtime context used to select the active chat model."""

    provider: str
    model: str
    thread_id: str | None = None
    user_id: str | None = None


def _provider_model_from_context(context: Any) -> tuple[str | None, str | None]:
    """Extract provider/model from LangChain runtime context."""

    if context is None:
        return None, None
    if isinstance(context, dict):
        provider = context.get("provider")
        model = context.get("model")
    else:
        provider = getattr(context, "provider", None)
        model = getattr(context, "model", None)
    return str(provider) if provider else None, str(model) if model else None


def _get_provider_model_from_request(request: ModelRequest) -> tuple[str | None, str | None]:
    """Resolve provider/model for the current model call."""

    return _provider_model_from_context(getattr(request.runtime, "context", None))


class RuntimeModelMiddleware(AgentMiddleware):
    """Swap the active LLM per request using runtime context."""

    def _override_request_model(self, request: ModelRequest) -> ModelRequest:
        provider, model = _get_provider_model_from_request(request)
        if not provider or not model:
            return request

        try:
            # build_chat_model caches on (provider, model, temperature, overrides),
            # so this does not rebuild a connection pool on every model call.
            llm = build_chat_model(
                provider=provider,
                model=model,
                temperature=0.0,
                streaming=bool(getattr(request.model, "streaming", False)),
            )
        except Exception:
            logger.exception(
                "Failed to create runtime LLM for provider=%s model=%s",
                provider,
                model,
            )
            raise

        logger.debug(
            "Using runtime LLM provider=%s model=%s",
            provider,
            model,
        )
        return request.override(model=llm)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Select the model for the current sync request."""

        return handler(self._override_request_model(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Select the model for the current async request."""

        return await handler(self._override_request_model(request))
