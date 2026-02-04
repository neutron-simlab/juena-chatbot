"""
LLM Provider Factory and Utilities
Centralized LLM provider management for OpenAI, Blablador, and future providers
"""
from typing import Dict
from langchain_openai import ChatOpenAI
from juena.schema.llm_models import BlabladorModelName, Provider


def _get_config():
    """Lazy import of global_config to avoid circular import"""
    from juena.core.config import global_config
    return global_config


class LLMFactory:
    """Factory for creating LLM instances with different providers"""
    
    @staticmethod
    def create_llm(
        provider: str = None,
        model: str = None,
        temperature: float = 0.0,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create LLM instance based on provider
        
        Args:
            provider: 'openai' and 'blablador'
            model: Model name (provider-specific)
            temperature: Temperature setting
            **kwargs: Provider-specific parameters
        
        Returns:
            ChatOpenAI instance
        """
        
        # Use config defaults if not specified
        config = _get_config()
        provider = provider or config.DEFAULT_PROVIDER
        model = model or config.DEFAULT_MODEL
        
        if provider.lower() == 'openai':
            return LLMFactory._create_openai(model, temperature, **kwargs)
        elif provider.lower() == 'blablador':
            return LLMFactory._create_blablador(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Available: openai and blablador")
    
    @staticmethod
    def _create_openai(model: str, temperature: float, **kwargs) -> ChatOpenAI:
        """Create OpenAI LLM"""
        config = _get_config()
        llm_kwargs = {
            'api_key': config.OPENAI_API_KEY,
            'model': model,
            'temperature': temperature,
            'max_tokens': kwargs.get('max_tokens', config.MAX_TOKENS),
            'timeout': kwargs.get('timeout', config.TIMEOUT_SECONDS),
            'max_retries': kwargs.get('max_retries', config.MAX_RETRIES),
        }
        # Only set streaming if explicitly provided in kwargs
        if 'streaming' in kwargs:
            llm_kwargs['streaming'] = kwargs['streaming']
        return ChatOpenAI(**llm_kwargs)
    
    @staticmethod
    def _create_blablador(model: str, temperature: float, **kwargs) -> ChatOpenAI:
        """Create Blablador LLM (uses ChatOpenAI with custom base_url)
        
        Note: timeout is properly configured here to prevent hanging.
        The timeout value (default 60s from config.TIMEOUT_SECONDS) 
        is passed to ChatOpenAI which will raise a timeout error if exceeded.
        """
        config = _get_config()
        timeout = kwargs.get('timeout', config.TIMEOUT_SECONDS)
        llm_kwargs = {
            'api_key': config.BLABLADOR_API_KEY,
            'base_url': config.BLABLADOR_BASE_URL,
            'model': model,
            'temperature': temperature,
            'max_tokens': kwargs.get('max_tokens', config.MAX_TOKENS),
            'timeout': timeout,  # Timeout in seconds - prevents hanging on Blablador
            'max_retries': kwargs.get('max_retries', config.MAX_RETRIES),
        }
        # Only set streaming if explicitly provided in kwargs
        if 'streaming' in kwargs:
            llm_kwargs['streaming'] = kwargs['streaming']
        return ChatOpenAI(**llm_kwargs)


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
    Check which providers are available (have API keys configured)
    
    This function delegates to Config.get_available_providers() to avoid code duplication.
    The actual implementation is in config.py to avoid circular imports.
    
    Returns:
        Dict mapping provider names to availability status
    """
    config = _get_config()
    return config.get_available_providers()


def get_available_models(provider: str) -> list[str]:
    """
    Get list of available models for a provider based on .env configuration.
    
    If provider-specific AVAILABLE_MODELS env var is set, returns that filtered list.
    Otherwise, returns all models from the enum for that provider.
    
    Args:
        provider: Provider name ('openai' or 'blablador')
        
    Returns:
        List of available model names for the provider
    """
    config = _get_config()
    return config.get_available_models(provider)


def validate_provider_config(provider: str) -> bool:
    """
    Validate that a specific provider is properly configured
    
    Args:
        provider: Provider name to validate
    
    Returns:
        True if provider is configured and available
    """
    available = get_available_providers()
    return available.get(provider.lower(), False)
