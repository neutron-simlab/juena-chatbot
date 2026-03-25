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

    for key in (
        "DB_DIR",
        "CHECKPOINT_DB_PATH",
        "CHAT_DB_PATH",
        "CONTEXT7_API_KEY",
        "CONTEXT7_MCP_URL",
        "CONTEXT7_TIMEOUT_SECONDS",
        "TAVILY_API_KEY",
    ):
        env.pop(key, None)

    if extra_env:
        env.update(extra_env)

    script = """
import json
from pathlib import Path
from juena.core.config import global_config

print("JSON::" + json.dumps({
    "db_dir": global_config.DB_DIR,
    "checkpoint_db_path": global_config.CHECKPOINT_DB_PATH,
    "chat_db_path": global_config.CHAT_DB_PATH,
    "db_dir_exists": Path(global_config.DB_DIR).is_dir(),
    "checkpoint_parent_exists": Path(global_config.CHECKPOINT_DB_PATH).parent.is_dir(),
    "chat_parent_exists": Path(global_config.CHAT_DB_PATH).parent.is_dir(),
    "context7_api_key": global_config.CONTEXT7_API_KEY,
    "context7_mcp_url": global_config.CONTEXT7_MCP_URL,
    "context7_timeout_seconds": global_config.CONTEXT7_TIMEOUT_SECONDS,
    "tavily_api_key": global_config.TAVILY_API_KEY,
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


def test_default_db_dir_is_repo_local() -> None:
    snapshot = _load_config_snapshot()

    expected = REPO_ROOT / "data" / "db"
    assert snapshot["db_dir"] == str(expected)
    assert snapshot["checkpoint_db_path"] == str(expected / "checkpoints.sqlite")
    assert snapshot["chat_db_path"] == str(expected / "chats.sqlite")
    assert snapshot["db_dir_exists"] is True


def test_overridden_db_file_parents_are_created(tmp_path: Path) -> None:
    checkpoint_db = tmp_path / "custom-checkpoints" / "checkpoints.sqlite"
    chat_db = tmp_path / "custom-history" / "chats.sqlite"

    snapshot = _load_config_snapshot(
        {
            "DB_DIR": str(tmp_path / "db-root"),
            "CHECKPOINT_DB_PATH": str(checkpoint_db),
            "CHAT_DB_PATH": str(chat_db),
        }
    )

    assert snapshot["checkpoint_db_path"] == str(checkpoint_db)
    assert snapshot["chat_db_path"] == str(chat_db)
    assert snapshot["checkpoint_parent_exists"] is True
    assert snapshot["chat_parent_exists"] is True


def test_context7_defaults_are_parsed() -> None:
    snapshot = _load_config_snapshot()

    assert snapshot["context7_api_key"] is None
    assert snapshot["context7_mcp_url"] == "https://mcp.context7.com/mcp"
    assert snapshot["context7_timeout_seconds"] == 30
    assert snapshot["tavily_api_key"] is None
