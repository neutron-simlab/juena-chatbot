"""Contracts enforced by the development launcher itself."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


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
               JUENA_RAG_REPO=str(tmp_path / "missing-rag"),
               PROD_PROXY_MODE="auto", PROD_EDGE_DIR=str(tmp_path / "edge"))
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


def _production_launcher(tmp_path: Path, *, shared: bool = True):
    env = _launcher(tmp_path)
    helper = tmp_path / "deploy/sandbox-production.sh"
    helper.parent.mkdir()
    helper.write_text('''
production_worker_installed() { [ "${TEST_WORKER_INSTALLED:-false}" = true ]; }
production_worker_stop() {
    if production_worker_installed; then echo worker-stop >> "$COMMAND_LOG"; fi
}
production_worker_start() { echo worker-start >> "$COMMAND_LOG"; }
''')
    config = tmp_path / "server.env"
    config.write_text("POSTGRES_PASSWORD=test\nRAG_POSTGRES_PASSWORD=test\n")
    env["PROD_ENV_FILE"] = str(config)
    if shared:
        edge = tmp_path / "edge"
        edge.mkdir()
        for name in ("compose.yml", "compose.juena.yml", "nginx.with-juena.conf"):
            (edge / name).touch()
    (tmp_path / "bin/docker").write_text('''#!/bin/sh
printf '%s\n' "$*" >> "$COMMAND_LOG"
case "$*" in
  *'config --environment')
    printf 'ENVIRONMENT=production\nAUTH_DEV_BYPASS=%s\nJUENA_PUBLIC_URL=https://juena.example\nSANDBOX_ENABLED=%s\n' "${TEST_BYPASS:-false}" "${TEST_SANDBOX:-false}" ;;
  *'ps -q edge') echo test-edge ;;
  'inspect '*) printf '%s/compose.yml,%s/compose.metadata.yml\n' "$PROD_EDGE_DIR" "$PROD_EDGE_DIR" ;;
  *'up -d --build '*) exit "${TEST_APP_FAILURE:-0}" ;;
  *'run --rm --no-deps edge nginx -t') exit "${TEST_NGINX_FAILURE:-0}" ;;
esac
''')
    for name in ("curl", "sleep"):
        executable = tmp_path / "bin" / name
        executable.write_text('''#!/bin/sh
printf '%s %s\n' "${0##*/}" "$*" >> "$COMMAND_LOG"
case "${0##*/}" in curl) exit "${TEST_HTTPS_FAILURE:-0}" ;; esac
''')
        executable.chmod(0o755)
    return env


def _production_run(tmp_path: Path, env: dict, *args: str):
    result = subprocess.run([str(tmp_path / "juena"), *args], cwd="/",
                            env=env, capture_output=True, text=True)
    commands = (tmp_path / "commands").read_text()
    assert "docker-compose.dev.yml" not in commands
    assert ".sandbox" not in commands
    return result, commands.splitlines()


def test_shared_up_waits_for_app_before_activating_gateway(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode == 0, result.stderr
    app = next(i for i, line in enumerate(commands) if "up -d --build" in line)
    check = next(i for i, line in enumerate(commands) if "nginx -t" in line)
    gateway = next(i for i, line in enumerate(commands) if "up -d --no-deps" in line)
    assert app < check < gateway
    assert commands[app].endswith("--wait-timeout 900 juena-chatbot")
    assert "-p juena-chatbot -f docker-compose.yml -f docker-compose.shared.yml" in commands[app]
    assert all("--profile server" not in line for line in commands)
    assert f"Deploying from {tmp_path}" in result.stdout
    assert any("https://juena.example/_stcore/health" in line for line in commands)


def test_failed_app_start_keeps_shared_gateway_unchanged(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    env["TEST_APP_FAILURE"] = "1"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode != 0
    assert not any("up -d --no-deps" in line for line in commands)
    assert "configuration was not changed" in result.stderr


def test_failed_nginx_preflight_keeps_shared_gateway_unchanged(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    env["TEST_NGINX_FAILURE"] = "1"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode != 0
    assert not any("up -d --no-deps" in line for line in commands)


def test_failed_https_restores_previous_gateway_configuration(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    env["TEST_HTTPS_FAILURE"] = "1"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode != 0
    changes = [line for line in commands if "up -d --no-deps" in line]
    assert len(changes) == 2
    assert "compose.juena.yml" in changes[0]
    assert "compose.metadata.yml" in changes[1]


def test_shared_down_preserves_independent_gateway_and_volumes(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    result, commands = _production_run(tmp_path, env, "prod", "down")
    assert result.returncode == 0, result.stderr
    assert commands[-1].endswith(" down")
    assert "-p juena-chatbot" in commands[-1]
    assert not any("-p juena-edge" in line or "--volumes" in line for line in commands)


def test_standalone_up_keeps_legacy_command_working(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path, shared=False)
    result, commands = _production_run(tmp_path, env, "prod-up")
    assert result.returncode == 0, result.stderr
    assert any("--profile server" in line and "up -d --build" in line for line in commands)
    assert not any("-p juena-edge" in line for line in commands)


def test_explicit_standalone_ignores_an_existing_shared_gateway_directory(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path, shared=True)
    env["PROD_PROXY_MODE"] = "standalone"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode == 0, result.stderr
    assert any("--profile server" in line and "up -d --build" in line for line in commands)
    assert not any("docker-compose.shared.yml" in line or "-p juena-edge" in line for line in commands)


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_prod_up_rejects_development_bypass(tmp_path: Path, shared: bool) -> None:
    env = _production_launcher(tmp_path, shared=shared)
    env["TEST_BYPASS"] = "true"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode != 0
    assert not any("up -d" in line for line in commands)


def test_prod_config_is_quiet_by_default(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    result, commands = _production_run(tmp_path, env, "prod", "config")
    assert result.returncode == 0, result.stderr
    assert commands[-1].endswith("config --quiet")


def test_prod_doctor_uses_running_app_without_rebuild_or_secret_dump(tmp_path: Path) -> None:
    env = _production_launcher(tmp_path)
    diagnostic = tmp_path / "src/juena/deployment/doctor.py"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("print('diagnostic')\n")
    result, commands = _production_run(tmp_path, env, "prod", "doctor")
    assert result.returncode == 0, result.stderr
    assert commands[-1].endswith("exec -T juena-chatbot python -")
    assert "POSTGRES_PASSWORD" not in result.stdout
    assert not any("build" in command or " up " in command for command in commands)


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_enabled_sandbox_stops_before_deployment_and_starts_after_health(tmp_path: Path, shared: bool):
    env = _production_launcher(tmp_path, shared=shared)
    env.update(TEST_SANDBOX="true", TEST_WORKER_INSTALLED="true")
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode == 0, result.stderr
    deploy = next(i for i, line in enumerate(commands) if "up -d --build" in line)
    assert "--wait --wait-timeout 900" in commands[deploy]
    assert commands.index("worker-stop") < deploy < commands.index("worker-start")
    if shared:
        healthy = next(i for i, line in enumerate(commands) if "https://juena.example/_stcore/health" in line)
        assert deploy < healthy < commands.index("worker-start")
    else:
        assert "--profile server" in commands[deploy]
        assert not any("-p juena-edge" in line for line in commands)


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_enabled_sandbox_requires_installed_worker_before_deployment(tmp_path: Path, shared: bool):
    env = _production_launcher(tmp_path, shared=shared)
    env["TEST_SANDBOX"] = "true"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode != 0
    assert "sandbox-setup" in result.stderr
    assert not any("up -d" in line for line in commands)


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_disabling_execution_stops_the_installed_worker(tmp_path: Path, shared: bool):
    env = _production_launcher(tmp_path, shared=shared)
    env["TEST_WORKER_INSTALLED"] = "true"
    result, commands = _production_run(tmp_path, env, "prod", "up")
    assert result.returncode == 0, result.stderr
    assert "worker-stop" in commands
    assert "worker-start" not in commands


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_down_stops_the_worker_before_its_database(tmp_path: Path, shared: bool):
    env = _production_launcher(tmp_path, shared=shared)
    env["TEST_WORKER_INSTALLED"] = "true"
    result, commands = _production_run(tmp_path, env, "prod", "down")
    assert result.returncode == 0, result.stderr
    assert commands.index("worker-stop") < len(commands) - 1
    assert commands[-1].endswith(" down")
