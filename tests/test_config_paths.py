"""Tests for config path defaults and directory initialization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config_snapshot(extra_env: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["JUENA_ENV_PATH"] = str(REPO_ROOT / ".env.test.empty")
    env["OPENAI_API_KEY"] = "test-key"
    env["DEFAULT_PROVIDER"] = "openai"
    env["FALLBACK_PROVIDER"] = "openai"
    env["LANGSMITH_TRACING"] = "false"
    env["LANGFUSE_TRACING"] = "false"
    env["ENVIRONMENT"] = "development"
    env["AUTH_DEV_BYPASS"] = "false"

    for key in (
        "DATABASE_URL",
        "DEFAULT_MODEL",
        "OPENAI_DEFAULT_MODEL",
        "BLABLADOR_DEFAULT_MODEL",
        "CONTEXT7_API_KEY",
        "CONTEXT7_MCP_URL",
        "CONTEXT7_TIMEOUT_SECONDS",
        "TAVILY_API_KEY",
        "JUENA_API_INTERNAL_URL",
        "JUENA_PUBLIC_URL",
        "UI_PUBLIC_URL",
        "SAML_SP_ENTITY_ID",
        "SAML_SP_ACS_URL",
        "SAML_SP_SLS_URL",
        "SESSION_COOKIE_SECURE",
    ):
        env.pop(key, None)

    if extra_env:
        env.update(extra_env)

    script = """
import json
from pathlib import Path
from juena.core.config import global_config
from juena_core.sandbox.config import sandbox_settings

sandbox = sandbox_settings()

print("JSON::" + json.dumps({
    "database_url": global_config.DATABASE_URL,
    "blablador_default_model": global_config.BLABLADOR_DEFAULT_MODEL,
    "context7_api_key": global_config.CONTEXT7_API_KEY,
    "context7_mcp_url": global_config.CONTEXT7_MCP_URL,
    "context7_timeout_seconds": global_config.CONTEXT7_TIMEOUT_SECONDS,
    "tavily_api_key": global_config.TAVILY_API_KEY,
    "internal_api_url": global_config.JUENA_API_INTERNAL_URL,
    "public_url": global_config.JUENA_PUBLIC_URL,
    "ui_public_url": global_config.UI_PUBLIC_URL,
    "cookie_secure": global_config.SESSION_COOKIE_SECURE,
    "sandbox_concurrency": sandbox.concurrency,
    "sandbox_cpu_limit": sandbox.cpu_limit,
    "sandbox_memory_limit": sandbox.memory_limit,
    "sandbox_workspace_limit_bytes": sandbox.workspace_limit_bytes,
    "sandbox_idle_ttl_seconds": sandbox.idle_ttl_seconds,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    marker = "JSON::"
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(f"Config snapshot marker not found in stdout: {result.stdout}")


def test_postgres_is_not_silently_defaulted_outside_compose() -> None:
    snapshot = _load_config_snapshot()

    assert snapshot["database_url"] is None


def test_blablador_defaults_to_juwels_qwen() -> None:
    snapshot = _load_config_snapshot()

    assert snapshot["blablador_default_model"] == (
        "90 - Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster - "
        "An excellent model from September 2026"
    )


def test_blablador_default_model_can_be_overridden() -> None:
    snapshot = _load_config_snapshot({"BLABLADOR_DEFAULT_MODEL": "deployment-model"})

    assert snapshot["blablador_default_model"] == "deployment-model"


def test_postgres_url_is_taken_verbatim_from_the_environment() -> None:
    snapshot = _load_config_snapshot(
        {"DATABASE_URL": "postgresql://juena:test@localhost/juena"}
    )

    assert snapshot["database_url"] == "postgresql://juena:test@localhost/juena"


def test_context7_defaults_are_parsed() -> None:
    snapshot = _load_config_snapshot()

    assert snapshot["context7_api_key"] is None
    assert snapshot["context7_mcp_url"] == "https://mcp.context7.com/mcp"
    assert snapshot["context7_timeout_seconds"] == 30
    assert snapshot["tavily_api_key"] is None


def test_development_url_and_cookie_defaults_remain_local() -> None:
    snapshot = _load_config_snapshot()

    assert snapshot["internal_api_url"] == "http://localhost:8080"
    assert snapshot["public_url"] == "http://localhost:8080"
    assert snapshot["ui_public_url"] == "http://localhost:9501"
    assert snapshot["cookie_secure"] is False


def test_sandbox_api_uses_the_worker_limit_environment() -> None:
    snapshot = _load_config_snapshot(
        {
            "SANDBOX_CONCURRENCY": "7",
            "SANDBOX_CPU_LIMIT": "1.5",
            "SANDBOX_MEMORY_LIMIT": "3g",
            "SANDBOX_WORKSPACE_LIMIT_BYTES": "123456",
            "SANDBOX_IDLE_TTL_SECONDS": "789",
        }
    )

    assert snapshot["sandbox_concurrency"] == 7
    assert snapshot["sandbox_cpu_limit"] == "1.5"
    assert snapshot["sandbox_memory_limit"] == "3g"
    assert snapshot["sandbox_workspace_limit_bytes"] == 123456
    assert snapshot["sandbox_idle_ttl_seconds"] == 789


def test_production_requires_public_https_configuration() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "JUENA_ENV_PATH": str(REPO_ROOT / ".env.test.empty"),
            "OPENAI_API_KEY": "test-key",
            "DEFAULT_PROVIDER": "openai",
            "FALLBACK_PROVIDER": "openai",
            "LANGSMITH_TRACING": "false",
            "LANGFUSE_TRACING": "false",
            "ENVIRONMENT": "production",
            "AUTH_DEV_BYPASS": "false",
        }
    )
    for key in (
        "JUENA_PUBLIC_URL",
        "UI_PUBLIC_URL",
        "SAML_SP_ENTITY_ID",
        "SAML_SP_ACS_URL",
        "SAML_SP_SLS_URL",
        "SESSION_COOKIE_SECURE",
    ):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "-c", "from juena.core.config import global_config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "JUENA_PUBLIC_URL is required in production" in output
    assert "SAML_SP_ACS_URL is required in production" in output


def test_production_accepts_explicit_secure_public_origins() -> None:
    snapshot = _load_config_snapshot(
        {
            "ENVIRONMENT": "production",
            "JUENA_API_INTERNAL_URL": "http://api:8080",
            "JUENA_PUBLIC_URL": "https://juena.example.org",
            "SAML_SP_ENTITY_ID": "https://juena.example.org/auth/metadata",
            "SAML_SP_ACS_URL": "https://juena.example.org/auth/acs",
            "SAML_SP_SLS_URL": "https://juena.example.org/auth/sls",
            "SESSION_COOKIE_SECURE": "true",
        }
    )

    assert snapshot["internal_api_url"] == "http://api:8080"
    assert snapshot["public_url"] == "https://juena.example.org"
    assert snapshot["ui_public_url"] == "https://juena.example.org"
    assert snapshot["cookie_secure"] is True


def _run_config(extra_env: dict[str, str], code: str) -> subprocess.CompletedProcess:
    """Import the real config in a fresh process with exactly these switches."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("LANGSMITH_", "LANGCHAIN_", "LANGFUSE_")):
            env.pop(key)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "JUENA_ENV_PATH": str(REPO_ROOT / ".env.test.empty"),
            "OPENAI_API_KEY": "test-key",
            "DEFAULT_PROVIDER": "openai",
            "FALLBACK_PROVIDER": "openai",
            "ENVIRONMENT": "development",
            "AUTH_DEV_BYPASS": "false",
            "LANGSMITH_TRACING": "false",
            "LANGFUSE_TRACING": "false",
            **extra_env,
        }
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


_LANGSMITH_EFFECTIVE = (
    "from juena.core.config import global_config\n"
    "from langsmith.utils import tracing_is_enabled\n"
    "print('TRACING::' + str(tracing_is_enabled()))\n"
)


def test_langsmith_switch_overrides_stray_v2_variables() -> None:
    # The langsmith library reads both `_V2` names before LANGSMITH_TRACING;
    # left alone, these would trace while startup printed "disabled".
    result = _run_config(
        {"LANGSMITH_TRACING_V2": "true", "LANGCHAIN_TRACING_V2": "true"},
        _LANGSMITH_EFFECTIVE,
    )

    assert result.returncode == 0, result.stderr
    assert "TRACING::False" in result.stdout
    assert "LangSmith: disabled" in result.stdout


def test_langsmith_switch_still_turns_tracing_on() -> None:
    # The other half: the check above is only meaningful if the same probe
    # reports True when the switch is on.
    result = _run_config(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "test-langsmith-key",
            "LANGCHAIN_TRACING_V2": "false",
        },
        _LANGSMITH_EFFECTIVE,
    )

    assert result.returncode == 0, result.stderr
    assert "TRACING::True" in result.stdout
    assert "LangSmith: enabled" in result.stdout


def test_langfuse_tracing_requires_its_url_and_keys() -> None:
    result = _run_config(
        {"LANGFUSE_TRACING": "true"},
        "from juena.core.config import global_config",
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        assert f"{name} is required when LANGFUSE_TRACING is enabled" in output


def test_langfuse_tracing_refuses_the_sdk_off_switch() -> None:
    # The SDK's own switch would drop every trace while startup said enabled.
    result = _run_config(
        {
            "LANGFUSE_TRACING": "true",
            "LANGFUSE_BASE_URL": "http://127.0.0.1:9",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_TRACING_ENABLED": "false",
        },
        "from juena.core.config import global_config",
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "LANGFUSE_TRACING_ENABLED=false conflicts with LANGFUSE_TRACING=true" in output


def test_langfuse_tracing_starts_when_configured() -> None:
    result = _run_config(
        {
            "LANGFUSE_TRACING": "true",
            "LANGFUSE_BASE_URL": "http://127.0.0.1:9",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        },
        "from juena.core.config import global_config",
    )

    assert result.returncode == 0, result.stderr
    assert "Langfuse: enabled" in result.stdout


def test_filesystem_identity_secret_is_required_with_execution_disabled() -> None:
    for secret in ("", "short"):
        try:
            _load_config_snapshot({"SANDBOX_ENABLED": "false", "SANDBOX_IDENTITY_SECRET": secret})
        except subprocess.CalledProcessError as exc:
            assert "SANDBOX_IDENTITY_SECRET" in exc.stdout + exc.stderr
            assert "32 characters" in exc.stdout + exc.stderr
        else:
            raise AssertionError("An invalid filesystem secret was accepted")
    _load_config_snapshot({"SANDBOX_ENABLED": "false", "SANDBOX_IDENTITY_SECRET": "stable-secret-with-at-least-32-characters"})
