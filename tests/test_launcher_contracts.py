"""Contracts enforced by the development launcher itself."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_core_check_fails_when_checkout_is_missing(tmp_path: Path) -> None:
    """A missing checkout must not turn the boundary check into a green run."""

    env = os.environ.copy()
    env["JUENA_CORE_REPO"] = str(tmp_path / "missing-core")

    result = subprocess.run(
        [str(ROOT / "juena"), "check-imports"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot check the import direction" in result.stderr
    assert "pytest" not in result.stdout


def _launcher(tmp_path: Path):
    import shutil

    shutil.copy(ROOT / "juena", tmp_path / "juena")
    (tmp_path / ".env").write_text("LANGFUSE_TRACING=false\nSANDBOX_ENABLED=false\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("uv", "docker"):
        executable = bin_dir / command
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
        executable.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f"{bin_dir}:{env['PATH']}", COMMAND_LOG=str(tmp_path / "commands"),
               JUENA_CORE_REPO=str(tmp_path / "missing-core"),
               JUENA_RAG_REPO=str(tmp_path / "missing-rag"))
    return env


def test_unit_suite_does_not_require_sibling_checkouts(tmp_path: Path) -> None:
    env = _launcher(tmp_path)
    result = subprocess.run([str(tmp_path / "juena"), "test", "-k", "example"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "pytest -q" in (tmp_path / "commands").read_text()


def test_ingest_uses_compose_and_preserves_flag_values(tmp_path: Path) -> None:
    env = _launcher(tmp_path)
    result = subprocess.run([str(tmp_path / "juena"), "repos-ingest", "jscatter",
                             "--limit", "2", "--dry-run"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands").read_text()
    assert "run --rm --build rag-ingest jscatter --limit 2 --dry-run" in commands
    assert "bootstrap" not in commands


def test_server_launcher_uses_only_server_environment(tmp_path: Path) -> None:
    env = _launcher(tmp_path)
    config = tmp_path / "server.env"
    config.write_text("POSTGRES_PASSWORD=test\nRAG_POSTGRES_PASSWORD=test\n")
    env["PROD_ENV_FILE"] = str(config)
    result = subprocess.run([str(tmp_path / "juena"), "prod-config", "--quiet"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands").read_text()
    assert f"--profile server --env-file {config} config --quiet" in commands
    assert "docker-compose.dev.yml" not in commands
    assert "network create" not in commands
