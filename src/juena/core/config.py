import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


from juena.schema.llm_models import BlabladorModelName as _BlabladorModelName

# Keep the built-in default in step with the enum the UI filters against.
_BLABLADOR_DEFAULT_MODEL_ID = _BlabladorModelName.GPT_OSS.value


def _find_repo_root() -> Path:
    """Resolve repo root by walking up from this file until we find pyproject.toml."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent  # fallback to config's parent dir


def _repo_data_dir(*parts: str) -> Path:
    """Build a path inside the repository-local data directory."""
    return _find_repo_root() / "data" / Path(*parts)

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
    # Example: OPENAI_AVAILABLE_MODELS=gpt-4o-mini
    OPENAI_AVAILABLE_MODELS = os.getenv("OPENAI_AVAILABLE_MODELS")
    
    # Blablador (OpenAI-compatible API)
    BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
    BLABLADOR_BASE_URL = os.getenv("BLABLADOR_BASE_URL")
    BLABLADOR_DEFAULT_MODEL = os.getenv(
        "BLABLADOR_DEFAULT_MODEL",
        _BLABLADOR_DEFAULT_MODEL_ID,
    )
    BLABLADOR_EMBEDDING_MODEL = os.getenv("BLABLADOR_EMBEDDING_MODEL", "alias-qwen3-8b-embeddings")
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

    # Send full tool results to the UI instead of a one-line activity summary.
    # Off by default: payloads are large, the model has already consumed them,
    # and rendering them buries the answer. Useful when debugging tool output.
    STREAM_TOOL_PAYLOADS = os.getenv("STREAM_TOOL_PAYLOADS", "false").lower() == "true"

    # Context7 MCP (optional external library/framework documentation tools)
    CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY")
    CONTEXT7_MCP_URL = os.getenv("CONTEXT7_MCP_URL", "https://mcp.context7.com/mcp")
    CONTEXT7_TIMEOUT_SECONDS = int(os.getenv("CONTEXT7_TIMEOUT_SECONDS", "30"))

    # Tavily web search (optional external web search tool)
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Sandboxed scientific execution. Disabled unless the API can reach the
    # separately deployed rootless-Podman broker with a shared secret.
    SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    # Development can show the starter for UI review without claiming that a
    # command can run. The click is intercepted until SANDBOX_ENABLED is true.
    SANDBOX_STARTER_PREVIEW = os.getenv(
        "SANDBOX_STARTER_PREVIEW",
        "true" if os.getenv("ENVIRONMENT", "development").lower() != "production" else "false",
    ).lower() == "true"
    SANDBOX_BROKER_URL = os.getenv("SANDBOX_BROKER_URL")
    SANDBOX_BROKER_TOKEN = os.getenv("SANDBOX_BROKER_TOKEN")
    SANDBOX_BROKER_TIMEOUT_SECONDS = int(
        os.getenv("SANDBOX_BROKER_TIMEOUT_SECONDS", "620")
    )
    SANDBOX_EXECUTION_TIMEOUT_SECONDS = int(
        os.getenv("SANDBOX_EXECUTION_TIMEOUT_SECONDS", "600")
    )
    SANDBOX_MAX_OUTPUT_BYTES = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "100000"))
    SANDBOX_ARTIFACT_ROOT = Path(
        os.getenv("SANDBOX_ARTIFACT_ROOT", str(_repo_data_dir("artifacts")))
    )
    SANDBOX_AUDIT_FILE = Path(
        os.getenv("SANDBOX_AUDIT_FILE", str(_repo_data_dir("audit", "sandbox.jsonl")))
    )
    
    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
    UI_PORT = int(os.getenv("UI_PORT", "9501"))
    
    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    _development = ENVIRONMENT.lower() != "production"

    # Streamlit talks to FastAPI over this server-side URL. Browsers must never
    # be sent here because production commonly uses private Compose DNS.
    JUENA_API_INTERNAL_URL = os.getenv(
        "JUENA_API_INTERNAL_URL",
        "http://localhost:8080",
    )
    # Browser-facing origin used to construct authentication links. Production
    # deliberately has no localhost default and is validated at startup.
    JUENA_PUBLIC_URL = os.getenv(
        "JUENA_PUBLIC_URL",
        "http://localhost:8080" if _development else None,
    )
    # Compatibility alias for one release. It remains useful in development,
    # where FastAPI and Streamlit run on separate localhost ports. A production
    # gateway serves both from JUENA_PUBLIC_URL, so this defaults to that origin.
    UI_PUBLIC_URL = os.getenv(
        "UI_PUBLIC_URL",
        "http://localhost:9501" if _development else JUENA_PUBLIC_URL,
    )
    
    # Application log directory (for date-based log files)
    LOG_DIR = os.getenv("LOG_DIR", "/tmp/logs")
    
    # Postgres is required by the authenticated production service. Imports and
    # unit tests remain possible without it; the FastAPI lifespan fails clearly.
    DATABASE_URL = os.getenv("DATABASE_URL")
    # Upper bound for the psycopg pool shared by the LangGraph checkpointer
    # and Store. SQLAlchemy manages its own pool for the application tables.
    DATABASE_POOL_MAX_SIZE = int(os.getenv("DATABASE_POOL_MAX_SIZE", "20"))

    # SAML service-provider settings.
    SAML_IDP_METADATA_URL = os.getenv(
        "SAML_IDP_METADATA_URL",
        "https://ifflogin.fz-juelich.de/saml/idp.xml",
    )
    SAML_SP_ENTITY_ID = os.getenv(
        "SAML_SP_ENTITY_ID",
        "http://localhost:8080/auth/metadata" if _development else None,
    )
    SAML_SP_ACS_URL = os.getenv(
        "SAML_SP_ACS_URL",
        "http://localhost:8080/auth/acs" if _development else None,
    )
    SAML_SP_SLS_URL = os.getenv(
        "SAML_SP_SLS_URL",
        "http://localhost:8080/auth/sls" if _development else None,
    )
    SAML_SP_CERT_FILE = os.getenv(
        "SAML_SP_CERT_FILE",
        str(_repo_data_dir("saml", "sp.crt")) if _development else None,
    )
    SAML_SP_KEY_FILE = os.getenv(
        "SAML_SP_KEY_FILE",
        str(_repo_data_dir("saml", "sp.key")) if _development else None,
    )
    XMLSEC_BINARY = os.getenv("XMLSEC_BINARY", "/usr/bin/xmlsec1")

    # Development-only authentication bypass, for use while SP registration
    # with the FZJ IdP is pending. Refused outright outside development: the
    # process will not start rather than silently exposing an unauthenticated
    # login route. See SAML-IFFLOGIN.md.
    _auth_dev_bypass = os.getenv("AUTH_DEV_BYPASS", "false").lower() == "true"
    if _auth_dev_bypass and not _development:
        raise RuntimeError(
            "AUTH_DEV_BYPASS cannot be enabled when ENVIRONMENT is production. "
            "It issues sessions without verifying any identity."
        )
    AUTH_DEV_BYPASS = _auth_dev_bypass
    SESSION_COOKIE_SECURE = os.getenv(
        "SESSION_COOKIE_SECURE",
        "false" if _development else "true",
    ).lower() == "true"
    SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "8"))

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
    def _default_model_for(cls, provider: str) -> str:
        """Provider-specific default model, matching the class-body derivation."""
        if provider.lower() == "blablador":
            return cls.BLABLADOR_DEFAULT_MODEL
        return cls.OPENAI_DEFAULT_MODEL

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
                # DEFAULT_MODEL was derived from the *original* provider at class
                # definition time. Re-derive it, or the pair desyncs into
                # something like provider=blablador with model=gpt-4o-mini.
                cls.DEFAULT_MODEL = cls._default_model_for(cls.DEFAULT_PROVIDER)
                print(
                    f"⚠️ DEFAULT_PROVIDER was unavailable, auto-adjusted to: "
                    f"{cls.DEFAULT_PROVIDER} (model: {cls.DEFAULT_MODEL})"
                )
            else:
                errors.append(
                    f"DEFAULT_PROVIDER '{cls.DEFAULT_PROVIDER}' is not available and no other providers are configured. "
                    f"Please configure at least one provider (OpenAI, Blablador, etc.) with valid API keys."
                )

        # Validate FALLBACK_PROVIDER if set
        if cls.FALLBACK_PROVIDER:
            fallback_available = available_providers.get(cls.FALLBACK_PROVIDER.lower(), False)
            if not fallback_available:
                # Read the name before clearing it: the old code interpolated
                # after the assignment and so always printed 'None'.
                unavailable = cls.FALLBACK_PROVIDER
                cls.FALLBACK_PROVIDER = None
                print(f"⚠️ FALLBACK_PROVIDER '{unavailable}' is not available, cleared fallback")
        
        if cls.LANGSMITH_TRACING and not cls.LANGSMITH_API_KEY:
            errors.append("LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled")

        if cls.SANDBOX_ENABLED:
            if not cls.SANDBOX_BROKER_URL:
                errors.append("SANDBOX_BROKER_URL is required when SANDBOX_ENABLED is true")
            if not cls.SANDBOX_BROKER_TOKEN:
                errors.append("SANDBOX_BROKER_TOKEN is required when SANDBOX_ENABLED is true")
            elif len(cls.SANDBOX_BROKER_TOKEN) < 32:
                errors.append("SANDBOX_BROKER_TOKEN must contain at least 32 characters")
            if cls.SANDBOX_EXECUTION_TIMEOUT_SECONDS <= 0:
                errors.append("SANDBOX_EXECUTION_TIMEOUT_SECONDS must be positive")
            if cls.SANDBOX_MAX_OUTPUT_BYTES <= 0:
                errors.append("SANDBOX_MAX_OUTPUT_BYTES must be positive")

        if not cls._development and cls.SANDBOX_STARTER_PREVIEW and not cls.SANDBOX_ENABLED:
            errors.append(
                "SANDBOX_STARTER_PREVIEW cannot be enabled in production without SANDBOX_ENABLED"
            )

        if not cls._development:
            if not cls.JUENA_PUBLIC_URL:
                errors.append("JUENA_PUBLIC_URL is required in production")
            elif not cls.JUENA_PUBLIC_URL.lower().startswith("https://"):
                errors.append("JUENA_PUBLIC_URL must use HTTPS in production")
            if not cls.UI_PUBLIC_URL:
                errors.append("UI_PUBLIC_URL is required in production")
            elif not cls.UI_PUBLIC_URL.lower().startswith("https://"):
                errors.append("UI_PUBLIC_URL must use HTTPS in production")
            for setting_name in (
                "SAML_SP_ENTITY_ID",
                "SAML_SP_ACS_URL",
                "SAML_SP_SLS_URL",
            ):
                value = getattr(cls, setting_name)
                if not value:
                    errors.append(f"{setting_name} is required in production")
                elif not value.lower().startswith("https://") or "localhost" in value.lower():
                    errors.append(
                        f"{setting_name} must use a public HTTPS URL in production"
                    )
            if not cls.SESSION_COOKIE_SECURE:
                errors.append("SESSION_COOKIE_SECURE must be true in production")
        
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
        print("✅ Persistence: Postgres (required at service startup)")
        
        return cls

# Initialize configuration on import
global_config = Config.initialize()
