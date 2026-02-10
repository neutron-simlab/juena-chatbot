import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


def _find_repo_root() -> Path:
    """Resolve repo root by walking up from this file until we find pyproject.toml."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent  # fallback to config's parent dir

# Load environment variables from .env file
# If JUENA_ENV_PATH is set (e.g. in Docker), use it; otherwise use <repo_root>/.env
# so that "cp env.example .env" in the repo root works without extra configuration.
path_env = os.getenv("JUENA_ENV_PATH")
if path_env is None or path_env == "":
    path_env = str(_find_repo_root() / ".env")
# Note: load_dotenv() by default does NOT override existing environment variables
# Use override=False to respect system env vars (default behavior)
load_dotenv(path_env, override=False)

class Config:
    """Essential configuration for JueNA"""
    
    # =============================================================================
    # REQUIRED SETTINGS
    # =============================================================================
    
    # OpenAI API Key
    # Check both .env file and system environment
    # Note: System environment variables take precedence over .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    
    # =============================================================================
    # LANGSMITH SETTINGS (Optional but recommended)
    # =============================================================================
    
    # LangSmith tracing configuration
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
    
    # =============================================================================
    # LLM PROVIDER CONFIGURATION
    # =============================================================================
    
    # Provider selection
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
    FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "openai")
    
    # Provider-specific default models
    # OpenAI (OpenAI API)
    OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    # Optional: Comma-separated list of available OpenAI models to show in UI
    # If not set, all models from OpenAIModelName enum will be shown
    # Example: OPENAI_AVAILABLE_MODELS=gpt-4o-mini,gpt-4o
    OPENAI_AVAILABLE_MODELS = os.getenv("OPENAI_AVAILABLE_MODELS")
    
    # Blablador (OpenAI-compatible API)
    BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
    BLABLADOR_BASE_URL = os.getenv("BLABLADOR_BASE_URL")
    BLABLADOR_DEFAULT_MODEL = os.getenv("BLABLADOR_DEFAULT_MODEL", "1 - GPT-OSS-120b - an open model released by OpenAI in August 2025")
    # Optional: Comma-separated list of available Blablador models to show in UI
    # If not set, all models from BlabladorModelName enum will be shown
    BLABLADOR_AVAILABLE_MODELS = os.getenv("BLABLADOR_AVAILABLE_MODELS")
    
    # Default model (uses provider-specific default based on DEFAULT_PROVIDER if not explicitly set)
    # If DEFAULT_MODEL is not set, it will use the provider-specific default (OPENAI_DEFAULT_MODEL or BLABLADOR_DEFAULT_MODEL)
    # based on DEFAULT_PROVIDER. This allows backward compatibility while supporting provider-specific defaults.
    _default_model_explicit = os.getenv("DEFAULT_MODEL")
    if _default_model_explicit:
        DEFAULT_MODEL = _default_model_explicit
    else:
        # Use provider-specific default based on DEFAULT_PROVIDER
        if DEFAULT_PROVIDER.lower() == "openai":
            DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
        elif DEFAULT_PROVIDER.lower() == "blablador":
            DEFAULT_MODEL = BLABLADOR_DEFAULT_MODEL
        else:
            # Fallback to OpenAI default for unknown providers
            DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "10000"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
    UI_PORT = int(os.getenv("UI_PORT", "8501"))
    
    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Application log directory (for date-based log files)
    LOG_DIR = os.getenv("LOG_DIR", "/tmp/logs")
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """
        Get dictionary of available providers (have API keys configured).
        
        This method checks all providers defined in the Provider enum dynamically,
        making it future-proof for new providers (Gemini, Anthropic, etc.).
        
        Returns:
            Dict mapping provider names to availability status
        """
        # Import Provider enum here to avoid circular import
        from juena.schema.llm_models import Provider
        
        providers = {}
        
        # Iterate over all Provider enum values (future-proof)
        for provider in Provider:
            provider_name = provider.value
            
            # Check provider-specific configuration requirements
            if provider_name == 'openai':
                # Check that API key exists and is not empty/whitespace
                has_api_key = bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.strip())
                providers[provider_name] = has_api_key
            elif provider_name == 'blablador':
                # Check that both API key and base URL exist and are not empty/whitespace
                has_config = bool(
                    cls.BLABLADOR_API_KEY and cls.BLABLADOR_API_KEY.strip() and
                    cls.BLABLADOR_BASE_URL and cls.BLABLADOR_BASE_URL.strip()
                )
                providers[provider_name] = has_config
            # Future providers can be added here:
            # elif provider_name == 'anthropic':
            #     providers[provider_name] = bool(cls.ANTHROPIC_API_KEY)
            # elif provider_name == 'gemini':
            #     providers[provider_name] = bool(cls.GEMINI_API_KEY)
            else:
                # Unknown provider - mark as unavailable
                providers[provider_name] = False
        
        return providers
    
    @classmethod
    def get_available_models(cls, provider: str) -> list[str]:
        """
        Get list of available models for a provider based on .env configuration.
        
        If provider-specific AVAILABLE_MODELS env var is set, returns that filtered list.
        Otherwise, returns all models from the enum for that provider.
        
        Args:
            provider: Provider name ('openai' or 'blablador')
            
        Returns:
            List of available model names for the provider
        """
        # Import here to avoid circular import
        from juena.schema.llm_models import (
            Provider,
            OpenAIModelName,
            BlabladorModelName,
            get_models_for_provider
        )
        
        try:
            provider_enum = Provider(provider.lower())
        except ValueError:
            return []
        
        # Get all models for this provider from enum
        all_models = get_models_for_provider(provider_enum)
        
        # Check if there's a filter configured in .env
        if provider.lower() == 'openai' and cls.OPENAI_AVAILABLE_MODELS:
            # Parse comma-separated list and filter
            configured_models = [
                model.strip() 
                for model in cls.OPENAI_AVAILABLE_MODELS.split(',')
                if model.strip()
            ]
            # Only return models that are both in enum and in configured list
            available_models = [
                model for model in all_models 
                if model in configured_models
            ]
            # If filter results in empty list, fall back to all models
            return available_models if available_models else all_models
        
        elif provider.lower() == 'blablador' and cls.BLABLADOR_AVAILABLE_MODELS:
            # Parse comma-separated list and filter
            configured_models = [
                model.strip() 
                for model in cls.BLABLADOR_AVAILABLE_MODELS.split(',')
                if model.strip()
            ]
            # Only return models that are both in enum and in configured list
            available_models = [
                model for model in all_models 
                if model in configured_models
            ]
            # If filter results in empty list, fall back to all models
            return available_models if available_models else all_models
        
        # No filter configured, return all models from enum
        return all_models
    
    @classmethod
    def validate_required(cls):
        """Validate that required environment variables are set"""
        errors = []
        
        # Get available providers
        available_providers = cls.get_available_providers()
        
        # Check if DEFAULT_PROVIDER is available
        default_provider_available = available_providers.get(cls.DEFAULT_PROVIDER.lower(), False)
        
        if not default_provider_available:
            # Auto-adjust to first available provider
            available_list = [p for p, available in available_providers.items() if available]
            if available_list:
                cls.DEFAULT_PROVIDER = available_list[0]
                print(f"⚠️ DEFAULT_PROVIDER was unavailable, auto-adjusted to: {cls.DEFAULT_PROVIDER}")
            else:
                errors.append(
                    f"DEFAULT_PROVIDER '{cls.DEFAULT_PROVIDER}' is not available and no other providers are configured. "
                    f"Please configure at least one provider (OpenAI, Blablador, etc.) with valid API keys."
                )
        
        # Validate FALLBACK_PROVIDER if set
        if cls.FALLBACK_PROVIDER:
            fallback_available = available_providers.get(cls.FALLBACK_PROVIDER.lower(), False)
            if not fallback_available:
                # Clear fallback if not available
                cls.FALLBACK_PROVIDER = None
                print(f"⚠️ FALLBACK_PROVIDER '{cls.FALLBACK_PROVIDER}' is not available, cleared fallback")
        
        if cls.LANGSMITH_TRACING and not cls.LANGSMITH_API_KEY:
            errors.append("LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled")
        
        if errors:
            raise ValueError(f"Missing required environment variables: {', '.join(errors)}")
    
    
    @classmethod
    def setup_langsmith(cls):
        """Setup LangSmith tracing if enabled"""
        if cls.LANGSMITH_TRACING and cls.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            if cls.LANGSMITH_ENDPOINT:
                os.environ["LANGSMITH_ENDPOINT"] = cls.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_API_KEY"] = cls.LANGSMITH_API_KEY
            if cls.LANGSMITH_PROJECT:
                os.environ["LANGSMITH_PROJECT"] = cls.LANGSMITH_PROJECT
            return True
        return False
    
    @classmethod
    def initialize(cls):
        """Initialize configuration - call once at startup"""
        cls.validate_required()
        langsmith_enabled = cls.setup_langsmith()
        
        # Get and display available providers
        available_providers = cls.get_available_providers()
        available_list = [p for p, available in available_providers.items() if available]
        
        print(f"✅ Configuration initialized")
        print(f"✅ Environment: {cls.ENVIRONMENT}")
        print(f"✅ Default Provider: {cls.DEFAULT_PROVIDER}")
        print(f"✅ Available Providers: {', '.join(available_list) if available_list else 'None'}")
        print(f"✅ LangSmith: {'enabled' if langsmith_enabled else 'disabled'}")
        
        return cls

# Initialize configuration on import
global_config = Config.initialize()
