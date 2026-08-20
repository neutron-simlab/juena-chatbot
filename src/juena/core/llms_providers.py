"""
LLM construction for the supported providers.

Chat models are built with :func:`langchain.chat_models.init_chat_model`, which
handles provider dispatch, credential wiring, and retry configuration. Blablador
exposes an OpenAI-compatible API, so both supported providers resolve through the
``openai`` provider and differ only by ``base_url``.

To add a provider that is *not* OpenAI-compatible, add it to the Provider enum in
llm_models.py, add a branch to :func:`_model_kwargs` supplying its credentials,
and pass the matching ``model_provider`` to ``init_chat_model``.
"""
from functools import lru_cache
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from juena.core.paths import get_config
from juena.schema.llm_models import (
    BLABLADOR_MAX_MODEL_LEN,
    Provider,
    get_models_for_provider,
    get_default_model_for_provider,
)


# Providers this build knows how to construct. All are OpenAI-compatible today.
_PROVIDERS: tuple[str, ...] = (Provider.OPENAI.value, Provider.BLABLADOR.value)


def _is_available(provider: str) -> bool:
    """Whether a provider has the credentials it needs to be constructed."""
    config = get_config()
    if provider == Provider.OPENAI.value:
        return bool((config.OPENAI_API_KEY or "").strip())
    if provider == Provider.BLABLADOR.value:
        # Blablador needs both a key and an endpoint; either alone is unusable.
        return bool(
            (config.BLABLADOR_API_KEY or "").strip()
            and (config.BLABLADOR_BASE_URL or "").strip()
        )
    return False


def get_available_providers() -> Dict[str, bool]:
    """
    Check which providers are available (have API keys configured).

    Returns:
        Dict mapping provider names to availability status
    """
    return {name: _is_available(name) for name in _PROVIDERS}


def _model_kwargs(provider: str, model: str, temperature: float, overrides: dict) -> dict:
    """Build the init_chat_model keyword arguments for a provider."""
    config = get_config()
    kwargs: dict[str, Any] = {
        # Every supported provider speaks the OpenAI Chat Completions API.
        "model_provider": Provider.OPENAI.value,
        "temperature": temperature,
        "max_tokens": overrides.get("max_tokens", config.MAX_TOKENS),
        "timeout": overrides.get("timeout", config.TIMEOUT_SECONDS),
        "max_retries": overrides.get("max_retries", config.MAX_RETRIES),
    }

    if provider == Provider.BLABLADOR.value:
        kwargs["api_key"] = config.BLABLADOR_API_KEY
        kwargs["base_url"] = config.BLABLADOR_BASE_URL
        context_window = BLABLADOR_MAX_MODEL_LEN.get(model)
        if context_window is not None:
            # What the model may *read*: the whole window minus the room already
            # reserved for what it writes. Summarization reads this to decide
            # when to compact, and an unset profile leaves it with no ceiling.
            kwargs["profile"] = {
                "max_input_tokens": context_window - kwargs["max_tokens"]
            }
    else:
        kwargs["api_key"] = config.OPENAI_API_KEY

    if "streaming" in overrides:
        kwargs["streaming"] = overrides["streaming"]

    return kwargs


def _build(provider: str, model: str, temperature: float, overrides: dict) -> BaseChatModel:
    return init_chat_model(model, **_model_kwargs(provider, model, temperature, overrides))


@lru_cache(maxsize=32)
def _build_cached(
    provider: str,
    model: str,
    temperature: float,
    override_items: tuple[tuple[str, Any], ...],
) -> BaseChatModel:
    """Cached construction. Each chat model owns an HTTP connection pool, so
    rebuilding one per model call (as the runtime model middleware does) is
    needlessly expensive."""
    return _build(provider, model, temperature, dict(override_items))


def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **overrides: Any,
) -> BaseChatModel:
    """
    Create a chat model for the given provider.

    Args:
        provider: Provider name (e.g. 'openai', 'blablador'); defaults to config
        model: Model name (provider-specific); defaults to config
        temperature: Temperature setting
        **overrides: max_tokens / timeout / max_retries / streaming overrides

    Returns:
        A configured BaseChatModel

    Raises:
        ValueError: if the provider is unknown
    """
    config = get_config()
    provider = (provider or config.DEFAULT_PROVIDER).lower()
    model = model or config.DEFAULT_MODEL

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {', '.join(_PROVIDERS)}"
        )

    override_items = tuple(sorted(overrides.items()))
    try:
        hash(override_items)
    except TypeError:
        # Unhashable override (e.g. a dict); skip the cache rather than fail.
        return _build(provider, model, temperature, overrides)
    return _build_cached(provider, model, temperature, override_items)


# Model discovery, read by the Streamlit sidebar and the /agents endpoint.
def get_available_models(provider: str) -> List[str]:
    """
    Get list of available models for a provider based on .env configuration.

    If provider-specific AVAILABLE_MODELS env var is set, returns that filtered list.
    Otherwise, returns all models from the enum for that provider.

    Args:
        provider: Provider name (e.g. 'openai', 'blablador')

    Returns:
        List of available model names for the provider
    """
    config = get_config()
    provider = provider.lower()

    try:
        all_models = get_models_for_provider(Provider(provider))
    except ValueError:
        return []

    env_filter = None
    if provider == Provider.OPENAI.value:
        env_filter = config.OPENAI_AVAILABLE_MODELS
    elif provider == Provider.BLABLADOR.value:
        env_filter = config.BLABLADOR_AVAILABLE_MODELS

    if env_filter:
        configured_models = [m.strip() for m in env_filter.split(',') if m.strip()]
        filtered = [m for m in all_models if m in configured_models]
        return filtered if filtered else all_models

    return all_models


def get_default_model(provider: str) -> str:
    """
    Get the default model for a provider.

    Args:
        provider: Provider name

    Returns:
        Default model name for the provider, or "" if the provider is unknown
    """
    try:
        return get_default_model_for_provider(Provider(provider.lower()))
    except ValueError:
        return ""


def format_model_name(provider: str, model: str) -> str:
    """
    Format a model name for UI display.

    Args:
        provider: Provider name
        model: Model ID

    Returns:
        Formatted model name for display
    """
    if provider.lower() == Provider.BLABLADOR.value:
        from juena.schema.llm_models import get_blablador_model_display_name
        return get_blablador_model_display_name(model)
    return model
