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
    env["ENVIRONMENT"] = "development"
    env["AUTH_DEV_BYPASS"] = "false"

    for key in (
        "DATABASE_URL",
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

print("JSON::" + json.dumps({
    "database_url": global_config.DATABASE_URL,
    "context7_api_key": global_config.CONTEXT7_API_KEY,
    "context7_mcp_url": global_config.CONTEXT7_MCP_URL,
    "context7_timeout_seconds": global_config.CONTEXT7_TIMEOUT_SECONDS,
    "tavily_api_key": global_config.TAVILY_API_KEY,
    "internal_api_url": global_config.JUENA_API_INTERNAL_URL,
    "public_url": global_config.JUENA_PUBLIC_URL,
    "ui_public_url": global_config.UI_PUBLIC_URL,
    "cookie_secure": global_config.SESSION_COOKIE_SECURE,
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
