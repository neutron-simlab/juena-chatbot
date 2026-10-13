import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

import juena_core
from juena_core.sandbox.config import SandboxRuntimeSettings, configure_sandbox

from juena.core.paths import repo_root


from juena_core.schema.llm_models import BlabladorModelName as _BlabladorModelName

# Keep the built-in default in step with the enum the UI filters against.
_BLABLADOR_DEFAULT_MODEL_ID = _BlabladorModelName.GPT_OSS.value


def _repo_data_dir(*parts: str) -> Path:
    """Build a path inside the repository-local data directory."""
    return repo_root() / "data" / Path(*parts)


# If JUENA_ENV_PATH is set (e.g. in Docker), use it; otherwise use <repo_root>/.env
# so that "cp env.example .env" in the repo root works without extra configuration.
path_env = os.getenv("JUENA_ENV_PATH")
if path_env is None or path_env == "":
    path_env = str(repo_root() / ".env")
# override=False so system environment variables win over the .env file.
load_dotenv(path_env, override=False)


class Config:
    """Essential configuration for JueNA"""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

    # Self-hosted Langfuse (deploy/langfuse/). Independent of LangSmith: with
    # both on, every run is traced to both. No default URL on purpose -- the
    # SDK's own default is Langfuse Cloud, and chats must never go there by
    # accident.
    LANGFUSE_TRACING = os.getenv("LANGFUSE_TRACING", "false").lower() == "true"
    LANGFUSE_BASE_URL = (os.getenv("LANGFUSE_BASE_URL") or "").strip() or None
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "blablador")
    FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "blablador")

    OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    # Comma-separated. Unset, every model in the OpenAIModelName enum is offered.
    OPENAI_AVAILABLE_MODELS = os.getenv("OPENAI_AVAILABLE_MODELS")

    # Blablador (OpenAI-compatible API)
    BLABLADOR_API_KEY = os.getenv("BLABLADOR_API_KEY")
    BLABLADOR_BASE_URL = os.getenv("BLABLADOR_BASE_URL")
    BLABLADOR_DEFAULT_MODEL = os.getenv(
        "BLABLADOR_DEFAULT_MODEL",
        _BLABLADOR_DEFAULT_MODEL_ID,
    )
    BLABLADOR_EMBEDDING_MODEL = os.getenv(
        "BLABLADOR_EMBEDDING_MODEL", "alias-qwen3-8b-embeddings"
    )
    # Comma-separated. Unset, every model in the BlabladorModelName enum is offered.
    BLABLADOR_AVAILABLE_MODELS = os.getenv("BLABLADOR_AVAILABLE_MODELS")

    # An explicit DEFAULT_MODEL still wins, which is what keeps older .env files working.
    _default_model_explicit = os.getenv("DEFAULT_MODEL")
    if _default_model_explicit:
        DEFAULT_MODEL = _default_model_explicit
    else:
        if DEFAULT_PROVIDER.lower() == "openai":
            DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
        elif DEFAULT_PROVIDER.lower() == "blablador":
            DEFAULT_MODEL = BLABLADOR_DEFAULT_MODEL
        else:
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

    # juena-rag, the shared retrieval service. Setting this is the switch: the
    # repository tools search pgvector over HTTP instead of the Chroma index
    # inside this pod. Unset, everything behaves exactly as before, which is what
    # makes the cutover reversible without a deploy.
    JUENA_RAG_URL = os.getenv("JUENA_RAG_URL", "").strip() or None

    # Sandboxed scientific execution. The `sandbox_jobs` table is the only
    # channel between the API and the host-side rootless-Podman worker.
    SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    # Development can show the starter for UI review without claiming that a
    # command can run. The click is intercepted until SANDBOX_ENABLED is true.
    SANDBOX_STARTER_PREVIEW = (
        os.getenv(
            "SANDBOX_STARTER_PREVIEW",
            "true"
            if os.getenv("ENVIRONMENT", "development").lower() != "production"
            else "false",
        ).lower()
        == "true"
    )
    SANDBOX_IDENTITY_SECRET = os.getenv("SANDBOX_IDENTITY_SECRET")
    SANDBOX_WORKSPACE_ROOT = Path(
        os.getenv("SANDBOX_WORKSPACE_ROOT", str(_repo_data_dir("sandbox-workspaces")))
    )
    # Execution slots. The API admits this many active jobs, so it must match the
    # worker's own SANDBOX_CONCURRENCY or the API will under- or over-admit.
    SANDBOX_CONCURRENCY = int(os.getenv("SANDBOX_CONCURRENCY", "4"))
    SANDBOX_EXECUTION_TIMEOUT_SECONDS = int(
        os.getenv("SANDBOX_EXECUTION_TIMEOUT_SECONDS", "600")
    )
    SANDBOX_MAX_OUTPUT_BYTES = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "100000"))
    # These four values are repeated by the host worker. The API uses them for
    # the approval card and workspace admission/cleanup, so reading the same
    # environment is what keeps what the user approves equal to what Podman
    # actually enforces.
    SANDBOX_CPU_LIMIT = os.getenv("SANDBOX_CPU_LIMIT", "2")
    SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "4g")
    SANDBOX_WORKSPACE_LIMIT_BYTES = int(
        os.getenv("SANDBOX_WORKSPACE_LIMIT_BYTES", str(5 * 1024**3))
    )
    SANDBOX_IDLE_TTL_SECONDS = int(
        os.getenv("SANDBOX_IDLE_TTL_SECONDS", str(24 * 3600))
    )
    SANDBOX_ARTIFACT_ROOT = Path(
        os.getenv("SANDBOX_ARTIFACT_ROOT", str(_repo_data_dir("artifacts")))
    )
    SANDBOX_AUDIT_FILE = Path(
        os.getenv("SANDBOX_AUDIT_FILE", str(_repo_data_dir("audit", "sandbox.jsonl")))
    )

    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
    UI_PORT = int(os.getenv("UI_PORT", "9501"))

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
    SESSION_COOKIE_SECURE = (
        os.getenv(
            "SESSION_COOKIE_SECURE",
            "false" if _development else "true",
        ).lower()
        == "true"
    )
    SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "8"))

    @classmethod
    def _check_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available based on config attributes.

        This is used during initialization before the registry is available.
        """
        provider_name = provider_name.lower()
        if provider_name == "openai":
            return bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.strip())
        elif provider_name == "blablador":
            return bool(
                cls.BLABLADOR_API_KEY
                and cls.BLABLADOR_API_KEY.strip()
                and cls.BLABLADOR_BASE_URL
                and cls.BLABLADOR_BASE_URL.strip()
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
        from juena_core.schema.llm_models import Provider

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

        available_providers = cls.get_available_providers()
        default_provider_available = available_providers.get(
            cls.DEFAULT_PROVIDER.lower(), False
        )

        if not default_provider_available:
            available_list = [
                p for p, available in available_providers.items() if available
            ]
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

        if cls.FALLBACK_PROVIDER:
            fallback_available = available_providers.get(
                cls.FALLBACK_PROVIDER.lower(), False
            )
            if not fallback_available:
                # Read the name before clearing it: the old code interpolated
                # after the assignment and so always printed 'None'.
                unavailable = cls.FALLBACK_PROVIDER
                cls.FALLBACK_PROVIDER = None
                print(
                    f"⚠️ FALLBACK_PROVIDER '{unavailable}' is not available, cleared fallback"
                )

        if cls.LANGSMITH_TRACING and not cls.LANGSMITH_API_KEY:
            errors.append(
                "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled"
            )

        if cls.LANGFUSE_TRACING:
            for setting_name in (
                "LANGFUSE_BASE_URL",
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
            ):
                if not getattr(cls, setting_name):
                    errors.append(
                        f"{setting_name} is required when LANGFUSE_TRACING is enabled"
                    )
            # The SDK has its own off switch. Left at false it would silently
            # drop every trace while startup reported Langfuse as enabled.
            if os.getenv("LANGFUSE_TRACING_ENABLED", "").strip().lower() == "false":
                errors.append(
                    "LANGFUSE_TRACING_ENABLED=false conflicts with "
                    "LANGFUSE_TRACING=true; remove LANGFUSE_TRACING_ENABLED"
                )

        if cls.SANDBOX_ENABLED:
            if not cls.DATABASE_URL:
                errors.append("DATABASE_URL is required when SANDBOX_ENABLED is true")
            if cls.SANDBOX_CONCURRENCY < 1:
                errors.append("SANDBOX_CONCURRENCY must be at least 1")
            if not cls.SANDBOX_IDENTITY_SECRET:
                errors.append(
                    "SANDBOX_IDENTITY_SECRET is required when SANDBOX_ENABLED is true"
                )
            elif len(cls.SANDBOX_IDENTITY_SECRET) < 32:
                errors.append(
                    "SANDBOX_IDENTITY_SECRET must contain at least 32 characters"
                )
            if cls.SANDBOX_EXECUTION_TIMEOUT_SECONDS <= 0:
                errors.append("SANDBOX_EXECUTION_TIMEOUT_SECONDS must be positive")
            if cls.SANDBOX_MAX_OUTPUT_BYTES <= 0:
                errors.append("SANDBOX_MAX_OUTPUT_BYTES must be positive")
            if cls.SANDBOX_WORKSPACE_LIMIT_BYTES <= 0:
                errors.append("SANDBOX_WORKSPACE_LIMIT_BYTES must be positive")
            if cls.SANDBOX_IDLE_TTL_SECONDS <= 0:
                errors.append("SANDBOX_IDLE_TTL_SECONDS must be positive")
            try:
                if float(cls.SANDBOX_CPU_LIMIT) <= 0:
                    raise ValueError
            except ValueError:
                errors.append("SANDBOX_CPU_LIMIT must be a positive number")
            if not cls.SANDBOX_MEMORY_LIMIT.strip():
                errors.append("SANDBOX_MEMORY_LIMIT must not be empty")

        if (
            not cls._development
            and cls.SANDBOX_STARTER_PREVIEW
            and not cls.SANDBOX_ENABLED
        ):
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
                elif (
                    not value.lower().startswith("https://")
                    or "localhost" in value.lower()
                ):
                    errors.append(
                        f"{setting_name} must use a public HTTPS URL in production"
                    )
            if not cls.SESSION_COOKIE_SECURE:
                errors.append("SESSION_COOKIE_SECURE must be true in production")

        if errors:
            raise ValueError(
                f"Missing required environment variables: {', '.join(errors)}"
            )

    @classmethod
    def setup_langsmith(cls):
        """Make LANGSMITH_TRACING the one switch that decides LangSmith tracing.

        The langsmith library reads `LANGSMITH_TRACING_V2` (then the legacy
        `LANGCHAIN_TRACING_V2`) before `LANGSMITH_TRACING`, so a stray `_V2`
        left in the environment would keep tracing on while startup printed
        "disabled". Writing `LANGSMITH_TRACING_V2` in both branches settles it.
        """
        if cls.LANGSMITH_TRACING and cls.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            if cls.LANGSMITH_ENDPOINT:
                os.environ["LANGSMITH_ENDPOINT"] = cls.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_API_KEY"] = cls.LANGSMITH_API_KEY
            if cls.LANGSMITH_PROJECT:
                os.environ["LANGSMITH_PROJECT"] = cls.LANGSMITH_PROJECT
            return True
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        return False

    @classmethod
    def initialize(cls):
        """Initialize configuration - call once at startup"""
        cls.validate_required()
        langsmith_enabled = cls.setup_langsmith()

        available_providers = cls.get_available_providers()
        available_list = [
            p for p, available in available_providers.items() if available
        ]

        print(f"✅ Configuration initialized")
        print(f"✅ Environment: {cls.ENVIRONMENT}")
        print(f"✅ Default Provider: {cls.DEFAULT_PROVIDER}")
        print(
            f"✅ Available Providers: {', '.join(available_list) if available_list else 'None'}"
        )
        print(f"✅ LangSmith: {'enabled' if langsmith_enabled else 'disabled'}")
        print(f"✅ Langfuse: {'enabled' if cls.LANGFUSE_TRACING else 'disabled'}")
        print("✅ Persistence: Postgres (required at service startup)")

        return cls


global_config = Config.initialize()


def _configure_core() -> None:
    """Hand the validated values to juena_core, once, at import time.

    **After `initialize()`, never before.** `validate_required()` may still
    move `DEFAULT_PROVIDER`, `DEFAULT_MODEL` and `FALLBACK_PROVIDER` when the
    configured provider has no credentials; configuring core first would cache
    models built from the values it rejected.

    This is the whole of 00-BOUNDARY.md decision 1 in practice: `Config` keeps
    its `os.getenv` calls and its `validate_required()`, which knows about
    SAML, production HTTPS and this deployment's sandbox, and core is handed a
    frozen value object instead of being taught to read an environment.
    """

    juena_core.configure(
        juena_core.CoreSettings(
            OPENAI_API_KEY=Config.OPENAI_API_KEY,
            BLABLADOR_API_KEY=Config.BLABLADOR_API_KEY,
            BLABLADOR_BASE_URL=Config.BLABLADOR_BASE_URL,
            MAX_TOKENS=Config.MAX_TOKENS,
            TIMEOUT_SECONDS=Config.TIMEOUT_SECONDS,
            MAX_RETRIES=Config.MAX_RETRIES,
            DEFAULT_PROVIDER=Config.DEFAULT_PROVIDER,
            DEFAULT_MODEL=Config.DEFAULT_MODEL,
            OPENAI_AVAILABLE_MODELS=Config.OPENAI_AVAILABLE_MODELS,
            BLABLADOR_AVAILABLE_MODELS=Config.BLABLADOR_AVAILABLE_MODELS,
            OPENAI_DEFAULT_MODEL=Config.OPENAI_DEFAULT_MODEL,
            BLABLADOR_DEFAULT_MODEL=Config.BLABLADOR_DEFAULT_MODEL,
            STREAM_TOOL_PAYLOADS=Config.STREAM_TOOL_PAYLOADS,
            DATABASE_URL=Config.DATABASE_URL,
            DATABASE_POOL_MAX_SIZE=Config.DATABASE_POOL_MAX_SIZE,
            SESSION_TTL_HOURS=Config.SESSION_TTL_HOURS,
            SESSION_COOKIE_SECURE=Config.SESSION_COOKIE_SECURE,
            FALLBACK_PROVIDER=Config.FALLBACK_PROVIDER,
            # Core names this after what it times, not after where it runs:
            # a specialist's execution tool, whichever backend serves it.
            EXECUTE_TIMEOUT_SECONDS=Config.SANDBOX_EXECUTION_TIMEOUT_SECONDS,
            ARTIFACT_ROOT=Config.SANDBOX_ARTIFACT_ROOT,
            AUDIT_FILE=Config.SANDBOX_AUDIT_FILE,
            LOG_LEVEL=Config.LOG_LEVEL,
            LOG_DIR=Path(Config.LOG_DIR),
            # Both are facts about this deployment, and both are true: the API
            # binds SERVER_HOST and Compose publishes 8080. They exist so core
            # can refuse to pair a fixed, unauthenticated principal with a
            # reachable API -- which is not this application, since every route
            # here authenticates a real SAML session.
            BIND_HOST=Config.SERVER_HOST,
            API_PUBLISHED=True,
        )
    )

    # Separate object, separate call: the sandbox is an optional core feature,
    # and its settings must not become fields every consumer of core carries.
    # Importing the `sandbox_jobs` model -- and so creating the table -- happens
    # inside this call, and only when it is enabled.
    configure_sandbox(
        SandboxRuntimeSettings(
            enabled=Config.SANDBOX_ENABLED,
            identity_secret=Config.SANDBOX_IDENTITY_SECRET,
            workspace_root=Config.SANDBOX_WORKSPACE_ROOT,
            concurrency=Config.SANDBOX_CONCURRENCY,
            execution_timeout_seconds=Config.SANDBOX_EXECUTION_TIMEOUT_SECONDS,
            max_output_bytes=Config.SANDBOX_MAX_OUTPUT_BYTES,
            workspace_limit_bytes=Config.SANDBOX_WORKSPACE_LIMIT_BYTES,
            idle_ttl_seconds=Config.SANDBOX_IDLE_TTL_SECONDS,
            cpu_limit=Config.SANDBOX_CPU_LIMIT,
            memory_limit=Config.SANDBOX_MEMORY_LIMIT,
        )
    )


_configure_core()
