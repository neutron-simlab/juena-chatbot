"""
LLM Provider Factory and Utilities
Centralized LLM provider management using a registry pattern for easy extensibility.

To add a new provider:
1. Add the provider to the Provider enum in llm_models.py
2. Register the provider using register_provider() with:
   - check_available: Function to check if provider is configured
   - create_llm: Function to create the LLM instance
   - get_models: Function to get available models (optional)
   - get_default_model: Function to get the default model (optional)
   - format_model_name: Function to format model names for UI (optional)
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from langchain_openai import ChatOpenAI
from juena.schema.llm_models import (
    Provider,
    get_models_for_provider,
    get_default_model_for_provider,
)

def _get_config():
    """Lazy import of Config class to avoid circular import"""
    from juena.core.config import Config
    return Config


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a registered provider."""
    name: str
    check_available: Callable[[], bool]
    create_llm: Callable[[str, float, dict], ChatOpenAI]
    get_available_models: Optional[Callable[[], List[str]]] = None
    get_default_model: Optional[Callable[[], str]] = None
    format_model_name: Optional[Callable[[str], str]] = None


class ProviderRegistry:
    """Registry for LLM providers. Provides a single point of configuration."""
    
    _providers: Dict[str, ProviderConfig] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        check_available: Callable[[], bool],
        create_llm: Callable[[str, float, dict], ChatOpenAI],
        get_available_models: Optional[Callable[[], List[str]]] = None,
        get_default_model: Optional[Callable[[], str]] = None,
        format_model_name: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Register a new provider."""
        cls._providers[name.lower()] = ProviderConfig(
            name=name.lower(),
            check_available=check_available,
            create_llm=create_llm,
            get_available_models=get_available_models,
            get_default_model=get_default_model,
            format_model_name=format_model_name,
        )
    
    @classmethod
    def get(cls, name: str) -> Optional[ProviderConfig]:
        """Get a provider config by name."""
        return cls._providers.get(name.lower())
    
    @classmethod
    def get_all(cls) -> Dict[str, ProviderConfig]:
        """Get all registered providers."""
        return cls._providers.copy()
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a provider is available (configured with valid credentials)."""
        provider = cls.get(name)
        if provider is None:
            return False
        try:
            return provider.check_available()
        except Exception:
            return False
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Get dictionary of all providers and their availability status."""
        return {name: cls.is_available(name) for name in cls._providers}


# Convenience function for registration
def register_provider(
    name: str,
    check_available: Callable[[], bool],
    create_llm: Callable[[str, float, dict], ChatOpenAI],
    get_available_models: Optional[Callable[[], List[str]]] = None,
    get_default_model: Optional[Callable[[], str]] = None,
    format_model_name: Optional[Callable[[str], str]] = None,
) -> None:
    """Register a new LLM provider."""
    ProviderRegistry.register(
        name=name,
        check_available=check_available,
        create_llm=create_llm,
        get_available_models=get_available_models,
        get_default_model=get_default_model,
        format_model_name=format_model_name,
    )


# =============================================================================
# BUILT-IN PROVIDER IMPLEMENTATIONS
# =============================================================================

def _check_openai_available() -> bool:
    """Check if OpenAI is configured."""
    config = _get_config()
    return bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.strip())


def _create_openai_llm(model: str, temperature: float, kwargs: dict) -> ChatOpenAI:
    """Create OpenAI LLM instance."""
    config = _get_config()
    llm_kwargs = {
        'api_key': config.OPENAI_API_KEY,
        'model': model,
        'temperature': temperature,
        'max_tokens': kwargs.get('max_tokens', config.MAX_TOKENS),
        'timeout': kwargs.get('timeout', config.TIMEOUT_SECONDS),
        'max_retries': kwargs.get('max_retries', config.MAX_RETRIES),
    }
    if 'streaming' in kwargs:
        llm_kwargs['streaming'] = kwargs['streaming']
    return ChatOpenAI(**llm_kwargs)


def _check_blablador_available() -> bool:
    """Check if Blablador is configured."""
    config = _get_config()
    return bool(
        config.BLABLADOR_API_KEY and config.BLABLADOR_API_KEY.strip() and
        config.BLABLADOR_BASE_URL and config.BLABLADOR_BASE_URL.strip()
    )


def _create_blablador_llm(model: str, temperature: float, kwargs: dict) -> ChatOpenAI:
    """Create Blablador LLM instance (OpenAI-compatible API)."""
    config = _get_config()
    timeout = kwargs.get('timeout', config.TIMEOUT_SECONDS)
    llm_kwargs = {
        'api_key': config.BLABLADOR_API_KEY,
        'base_url': config.BLABLADOR_BASE_URL,
        'model': model,
        'temperature': temperature,
        'max_tokens': kwargs.get('max_tokens', config.MAX_TOKENS),
        'timeout': timeout,
        'max_retries': kwargs.get('max_retries', config.MAX_RETRIES),
    }
    if 'streaming' in kwargs:
        llm_kwargs['streaming'] = kwargs['streaming']
    return ChatOpenAI(**llm_kwargs)


def _get_blablador_display_name(model_id: str) -> str:
    """Format Blablador model name for UI display."""
    from juena.schema.llm_models import get_blablador_model_display_name
    return get_blablador_model_display_name(model_id)


# Register built-in providers
register_provider(
    name="openai",
    check_available=_check_openai_available,
    create_llm=_create_openai_llm,
    get_available_models=lambda: get_models_for_provider(Provider.OPENAI),
    get_default_model=lambda: get_default_model_for_provider(Provider.OPENAI),
)

register_provider(
    name="blablador",
    check_available=_check_blablador_available,
    create_llm=_create_blablador_llm,
    get_available_models=lambda: get_models_for_provider(Provider.BLABLADOR),
    get_default_model=lambda: get_default_model_for_provider(Provider.BLABLADOR),
    format_model_name=_get_blablador_display_name,
)


# =============================================================================
# LLM FACTORY (uses registry)
# =============================================================================

class LLMFactory:
    """Factory for creating LLM instances using the provider registry."""
    
    @staticmethod
    def create_llm(
        provider: str = None,
        model: str = None,
        temperature: float = 0.0,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create LLM instance based on provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'blablador')
            model: Model name (provider-specific)
            temperature: Temperature setting
            **kwargs: Provider-specific parameters
        
        Returns:
            ChatOpenAI instance
        """
        config = _get_config()
        provider = (provider or config.DEFAULT_PROVIDER).lower()
        model = model or config.DEFAULT_MODEL
        
        provider_config = ProviderRegistry.get(provider)
        if provider_config is None:
            available = list(ProviderRegistry.get_all().keys())
            raise ValueError(f"Unknown provider: {provider}. Available: {', '.join(available)}")
        
        return provider_config.create_llm(model, temperature, kwargs)


def create_llm_with_fallback(
    provider: str = None, 
    model: str = None, 
    temperature: float = 0.0,
    **kwargs
) -> ChatOpenAI:
    """
    Create LLM with automatic fallback to available providers.
    
    Fallback chain is built dynamically from available providers only,
    making it future-proof for new providers (Gemini, Anthropic, etc.).
    """
    
    # Use config defaults if not specified
    config = _get_config()
    provider = provider or config.DEFAULT_PROVIDER
    model = model or config.DEFAULT_MODEL
    
    try:
        return LLMFactory.create_llm(provider, model, temperature, **kwargs)
    except Exception as e:
        print(f"⚠️ {provider} failed: {e}")
        
        # Build fallback chain dynamically from available providers
        available_providers = get_available_providers()
        
        # Get list of available provider names (sorted for consistent fallback order)
        fallback_chain = [
            p.value for p in Provider 
            if available_providers.get(p.value, False) and p.value != provider.lower()
        ]
        
        if not fallback_chain:
            raise Exception(
                f"Provider {provider} failed and no fallback providers are available. "
                f"Please configure at least one provider with valid API keys."
            )
        
        for fallback_provider in fallback_chain:
            print(f"🔄 Trying fallback: {fallback_provider}")
            try:
                return LLMFactory.create_llm(fallback_provider, model, temperature, **kwargs)
            except Exception as fallback_error:
                print(f"❌ {fallback_provider} also failed: {fallback_error}")
                continue
        
        raise Exception(f"All available providers failed. Last error: {e}")


def get_available_providers() -> Dict[str, bool]:
    """
    Check which providers are available (have API keys configured).
    
    Uses the provider registry to check availability.
    
    Returns:
        Dict mapping provider names to availability status
    """
    return ProviderRegistry.get_available_providers()


def get_available_models(provider: str) -> List[str]:
    """
    Get list of available models for a provider based on .env configuration.
    
    If provider-specific AVAILABLE_MODELS env var is set, returns that filtered list.
    Otherwise, returns all models from the registry/enum for that provider.
    
    Args:
        provider: Provider name (e.g., 'openai', 'blablador')
        
    Returns:
        List of available model names for the provider
    """
    config = _get_config()
    provider_config = ProviderRegistry.get(provider.lower())
    
    # Get all models from registry if available, otherwise from enum
    if provider_config and provider_config.get_available_models:
        all_models = provider_config.get_available_models()
    else:
        all_models = []
    
    # Check for env-based filter
    env_filter = None
    if provider.lower() == 'openai':
        env_filter = config.OPENAI_AVAILABLE_MODELS
    elif provider.lower() == 'blablador':
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
        Default model name for the provider
    """
    provider_config = ProviderRegistry.get(provider.lower())
    if provider_config and provider_config.get_default_model:
        return provider_config.get_default_model()
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
    provider_config = ProviderRegistry.get(provider.lower())
    if provider_config and provider_config.format_model_name:
        return provider_config.format_model_name(model)
    return model
