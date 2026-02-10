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
    
    # =============================================================================
    # DATABASE CONFIGURATION (SQLite)
    # =============================================================================
    
    # Directory for SQLite databases (LangGraph checkpoints and chat history)
    # In Docker: /data/db, locally: <repo_root>/data/db
    DB_DIR = os.getenv("DB_DIR", "/data/db")
    
    # LangGraph checkpoint database path
    CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", os.path.join(DB_DIR, "checkpoints.sqlite"))
    
    # Chat history database path (for Streamlit sidebar)
    CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(DB_DIR, "chats.sqlite"))
    
    @classmethod
    def _check_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available based on config attributes.
        
        This is used during initialization before the registry is available.
        """
        provider_name = provider_name.lower()
        if provider_name == 'openai':
            return bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.strip())
        elif provider_name == 'blablador':
            return bool(
                cls.BLABLADOR_API_KEY and cls.BLABLADOR_API_KEY.strip() and
                cls.BLABLADOR_BASE_URL and cls.BLABLADOR_BASE_URL.strip()
            )
        return False
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """
        Get dictionary of available providers (have API keys configured).
        
        Uses direct config checks to avoid circular imports during initialization.
        
        Returns:
            Dict mapping provider names to availability status
        """
        from juena.schema.llm_models import Provider
        return {p.value: cls._check_provider_available(p.value) for p in Provider}
    
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
        
        # Ensure database directory exists
        db_dir = Path(cls.DB_DIR)
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Get and display available providers
        available_providers = cls.get_available_providers()
        available_list = [p for p, available in available_providers.items() if available]
        
        print(f"✅ Configuration initialized")
        print(f"✅ Environment: {cls.ENVIRONMENT}")
        print(f"✅ Default Provider: {cls.DEFAULT_PROVIDER}")
        print(f"✅ Available Providers: {', '.join(available_list) if available_list else 'None'}")
        print(f"✅ LangSmith: {'enabled' if langsmith_enabled else 'disabled'}")
        print(f"✅ Database directory: {cls.DB_DIR}")
        
        return cls

# Initialize configuration on import
global_config = Config.initialize()
