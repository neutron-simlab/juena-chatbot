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


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_reset_chat_builds_before_deleting_and_redeploys(tmp_path: Path, shared: bool):
    env = _production_launcher(tmp_path, shared=shared)
    env["TEST_WORKER_INSTALLED"] = "true"
    result, commands = _production_run(tmp_path, env, "prod", "reset-chat", "--confirm")
    assert result.returncode == 0, result.stderr
    build = next(i for i, c in enumerate(commands) if c.endswith("build juena-chatbot"))
    stop = next(i for i, c in enumerate(commands) if c.endswith("stop juena-chatbot"))
    drop = next(i for i, c in enumerate(commands) if " dropdb " in c)
    create = next(i for i, c in enumerate(commands) if " createdb " in c)
    up = next(i for i, c in enumerate(commands) if "up -d --build" in c)
    assert build < commands.index("worker-stop") < stop < drop < create < up
    assert commands[drop].endswith("dropdb --username=juena --maintenance-db=postgres --if-exists --force juena")
    assert not any(" down" in c or "volume rm" in c or "exec -T rag-db" in c for c in commands)


def test_reset_chat_requires_explicit_confirmation(tmp_path: Path):
    env = _production_launcher(tmp_path)
    result = subprocess.run([str(tmp_path / "juena"), "prod", "reset-chat"],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--confirm" in result.stderr
    assert not (tmp_path / "commands").exists()


@pytest.mark.parametrize("failure", ["build juena-chatbot", "pg_isready", "dropdb"])
def test_reset_chat_stops_on_failure(tmp_path: Path, failure: str):
    env = _production_launcher(tmp_path)
    docker = tmp_path / "bin/docker"
    docker.write_text(docker.read_text().replace(
        'case "$*" in', f'case "$*" in *"{failure}"*) exit 1 ;;', 1))
    result, commands = _production_run(tmp_path, env, "prod", "reset-chat", "--confirm")
    assert result.returncode != 0
    assert not any(" createdb " in c or "up -d --build" in c for c in commands)
    if failure != "dropdb":
        assert not any("stop juena-chatbot" in c or " dropdb " in c for c in commands)


@pytest.mark.parametrize("setting", ["POSTGRES_HOST=external", "POSTGRES_DB=postgres", "AUTH_DEV_BYPASS=true"])
def test_reset_chat_rejects_unsafe_targets_before_mutating(tmp_path: Path, setting: str):
    env = _production_launcher(tmp_path)
    docker = tmp_path / "bin/docker"
    docker.write_text(docker.read_text().replace(
        "printf 'ENVIRONMENT=production", f"printf '{setting}\\nENVIRONMENT=production", 1))
    result, commands = _production_run(tmp_path, env, "prod", "reset-chat", "--confirm")
    assert result.returncode != 0
    assert not any("build juena-chatbot" in c or "stop juena-chatbot" in c or " dropdb " in c for c in commands)


def _profile_setup(tmp_path: Path, values: dict, *, shared: bool = True):
    import json

    env = _production_launcher(tmp_path, shared=shared)
    serialized = {name: value.replace("$", "$$") for name, value in values.items()}
    env["PROFILE_VALUES"] = json.dumps({"services": {"settings": {"environment": serialized}}})
    docker = tmp_path / "bin/docker"
    docker.write_text(docker.read_text().replace('case "$*" in', '''case "$*" in
  *'config --format json') cat > "$PROFILE_TEMPLATE"; printf '%s' "$PROFILE_VALUES"; exit 0 ;;
  *'config --quiet') exit "${TEST_CONFIG_FAILURE:-0}" ;;
''', 1))
    env["PROFILE_TEMPLATE"] = str(tmp_path / "settings-template.yml")
    config = Path(env["PROD_ENV_FILE"])
    original = "POSTGRES_PASSWORD=test\nBLABLADOR_API_KEY='fake-old-rag'\nTAVILY_API_KEY='fake-old-tool'\nSANDBOX_IDENTITY_SECRET=stable\n"
    config.write_text(original)
    return env, config, original


@pytest.mark.parametrize("shared", [False, True], ids=["standalone", "shared"])
def test_profile_setup_prepares_secrets_without_deploying_or_printing_them(tmp_path: Path, shared: bool):
    import base64

    env, config, original = _profile_setup(tmp_path, {"BLABLADOR_API_KEY": "fake-old-rag"}, shared=shared)
    result, commands = _production_run(tmp_path, env, "prod", "profile-setup")
    assert result.returncode == 0, result.stderr
    updated = config.read_text()
    assert "RAG_BLABLADOR_API_KEY='fake-old-rag'" in updated
    assert "\nBLABLADOR_API_KEY=" not in updated and "TAVILY_API_KEY=" not in updated
    assert "SANDBOX_IDENTITY_SECRET=stable" in updated
    secret = next(line.split("=", 1)[1].strip("'") for line in updated.splitlines() if line.startswith("JUENA_SECRET_KEY="))
    assert len(base64.urlsafe_b64decode(secret)) == 32
    assert secret not in result.stdout + result.stderr and "fake-old-rag" not in result.stdout + result.stderr
    backup, = tmp_path.glob("server.env.before-profile.*")
    assert backup.read_text() == original
    assert config.stat().st_mode & 0o777 == backup.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("server.env.profile*"))
    assert not any(" up " in c or " stop " in c or "exec -T" in c for c in commands)
    assert any(("docker-compose.shared.yml" if shared else "--profile server") in c for c in commands)


def test_profile_setup_preserves_existing_secret_and_dedicated_rag_key_on_repeat(tmp_path: Path):
    values = {"JUENA_SECRET_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
              "RAG_BLABLADOR_API_KEY": "fake-dedicated-$token", "BLABLADOR_API_KEY": "fake-old-rag"}
    env, config, _ = _profile_setup(tmp_path, values)
    for _ in range(2):
        result, _ = _production_run(tmp_path, env, "prod", "profile-setup")
        assert result.returncode == 0, result.stderr
        assert f"JUENA_SECRET_KEY='{values['JUENA_SECRET_KEY']}'" in config.read_text()
        assert "RAG_BLABLADOR_API_KEY='fake-dedicated-$token'" in config.read_text()
        assert config.read_text().count("JUENA_SECRET_KEY=") == 1


@pytest.mark.parametrize("values", [{}, {"BLABLADOR_API_KEY": "fake", "JUENA_SECRET_KEY": "invalid"},
                                   {"BLABLADOR_API_KEY": "line\nbreak"}])
def test_profile_setup_refuses_missing_or_invalid_secrets_without_changing_file(tmp_path: Path, values: dict):
    env, config, original = _profile_setup(tmp_path, values)
    result, _ = _production_run(tmp_path, env, "prod", "profile-setup")
    assert result.returncode != 0
    assert config.read_text() == original
    assert not list(tmp_path.glob("server.env.profile*"))


def test_profile_setup_keeps_original_when_complete_compose_validation_fails(tmp_path: Path):
    env, config, original = _profile_setup(tmp_path, {"BLABLADOR_API_KEY": "fake-old-rag"})
    env["TEST_CONFIG_FAILURE"] = "1"
    result, _ = _production_run(tmp_path, env, "prod", "profile-setup")
    assert result.returncode != 0
    assert config.read_text() == original
    assert not list(tmp_path.glob("server.env.profile*"))


def test_local_profile_setup_migrates_only_local_env_without_deployment(tmp_path: Path):
    env, server, original = _profile_setup(tmp_path, {"BLABLADOR_API_KEY": "fake-old-rag"})
    local = tmp_path / ".env"
    local.write_text(original)
    result = subprocess.run([str(tmp_path / "juena"), "profile-setup"], cwd="/", env=env, capture_output=True, text=True)
    commands = (tmp_path / "commands").read_text().splitlines()
    assert result.returncode == 0, result.stderr
    assert "RAG_BLABLADOR_API_KEY='fake-old-rag'" in local.read_text()
    assert "JUENA_SECRET_KEY=" in local.read_text()
    assert server.read_text() == original
    backup, = tmp_path.glob(".env.before-profile.*")
    assert backup.read_text() == original
    assert local.stat().st_mode & 0o777 == backup.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".env.profile*"))
    assert "fake-old-rag" not in result.stdout + result.stderr
    assert not any(" up " in c or " stop " in c or "--profile server" in c for c in commands)


def test_development_up_validates_compose_before_build_or_start(tmp_path: Path):
    env = _launcher(tmp_path)
    docker = tmp_path / "bin/docker"
    docker.write_text(docker.read_text() + '\ncase "$*" in *"config --quiet"*) exit 1 ;; esac\n')
    result = subprocess.run([str(tmp_path / "juena"), "up"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    commands = (tmp_path / "commands").read_text().splitlines()
    assert len(commands) == 1 and commands[0].endswith("config --quiet")


def test_development_up_can_use_already_built_images(tmp_path: Path):
    env = _launcher(tmp_path)
    result = subprocess.run([str(tmp_path / "juena"), "up", "--no-build"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands").read_text().splitlines()
    assert commands[0].endswith("config --quiet")
    up, = [command for command in commands if " up " in command]
    assert "--no-build" in up and "--build" not in up.split()


def test_development_worker_survives_launcher_process_group_cleanup(tmp_path: Path):
    import json
    import shlex
    import signal

    source = (ROOT / "juena").read_text()
    start = source.index("start_sandbox_worker() {")
    end = source.index("\nstop_sandbox_worker()", start)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text('#!/usr/bin/env python3\nimport json,os,time\nprint(json.dumps({"pid":os.getpid(),"sid":os.getsid(0)}),flush=True)\ntime.sleep(60)\n')
    fake_uv.chmod(0o755)
    script = tmp_path / "start"
    runtime = tmp_path / "runtime"
    script.write_text('''#!/usr/bin/env bash
set -euo pipefail
SANDBOX_RUNTIME_DIR=''' + shlex.quote(str(runtime)) + '''
SANDBOX_WORKER_PID_FILE="$SANDBOX_RUNTIME_DIR/worker.pid"
SANDBOX_WORKER_LOG_FILE="$SANDBOX_RUNTIME_DIR/worker.log"
SANDBOX_WORKER_LIMITS_FILE="$SANDBOX_RUNTIME_DIR/limits"
worker_is_running() { [ -s "$SANDBOX_WORKER_PID_FILE" ] && kill -0 "$(cat "$SANDBOX_WORKER_PID_FILE")" 2>/dev/null; }
die() { echo "$*" >&2; exit 1; }
''' + source[start:end] + '\nstart_sandbox_worker\n')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    launcher = subprocess.Popen(["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    worker_pid = None
    try:
        stdout, stderr = launcher.communicate(timeout=10)
        assert launcher.returncode == 0, stdout + stderr
        worker_pid = int((runtime / "worker.pid").read_text())
        info = json.loads((runtime / "worker.log").read_text())
        assert info == {"pid": worker_pid, "sid": worker_pid}
        try:
            os.killpg(launcher.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.kill(worker_pid, 0)
    finally:
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait()


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
