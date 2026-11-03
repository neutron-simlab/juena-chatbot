"""Activation requires real execution evidence; local tests simulate the worker."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
import pytest

from juena.deployment.sandbox_check import PLOT_SCRIPT, check_worker
from juena_core.filesystem.storage import WorkspaceStore

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deploy/sandbox-production.sh"


@pytest.mark.parametrize("outcome", ["success", "missing", "invalid", "unreported", "failed", "unavailable"])
def test_smoke_test_requires_completed_job_valid_png_and_artifact_report(tmp_path, outcome):
    store = WorkspaceStore(tmp_path, limit_bytes=10_000_000, idle_ttl_seconds=3600)

    class Jobs:
        closed = False
        identity = None

        def open(self):
            pass

        def close(self):
            self.closed = True

        def worker_is_alive(self):
            return outcome != "unavailable"

        def execute(self, **kwargs):
            self.identity = kwargs["workspace_id"]
            assert kwargs["tenant_id"] == self.identity
            assert kwargs["command"] == "python /workspace/.scratch/smoke.py"
            script = store.workspace_dir(self.identity) / ".scratch/smoke.py"
            assert script.read_text() == PLOT_SCRIPT
            compile(script.read_text(), str(script), "exec")
            target = store.workspace_dir(self.identity) / "outputs/sandbox-smoke.png"
            if outcome == "invalid":
                target.write_bytes(b"not an image")
            elif outcome != "missing":
                Image.new("RGB", (10, 10)).save(target)
            return SimpleNamespace(
                status="failed" if outcome == "failed" else "completed",
                exit_code=1 if outcome == "failed" else 0,
                output="test execution",
                artifact_names=[] if outcome == "unreported" else ["sandbox-smoke.png"],
            )

    jobs = Jobs()
    assert check_worker(jobs, store, pause=lambda _: None) is (outcome == "success")
    assert jobs.closed
    if jobs.identity:
        assert store.workspace_root(jobs.identity).exists() is (outcome != "success")


def _shell(script, *args, env=None):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", script, "test", str(HELPER), *map(str, args)],
                          text=True, capture_output=True, env=env)


def test_worker_commands_leave_private_checkout_without_changing_launcher_cwd(tmp_path):
    checkout = tmp_path / "private-checkout"
    checkout.mkdir(mode=0o700)
    result = _shell('''
source "$1"
cd "$2"
id() { printf '1001\n'; }
runuser() {
    [ "$1" = -u ] && [ "$2" = juena-sandbox ] && [ "$3" = -- ] || return 1
    shift 3
    "$@"
}
production_worker_user bash -c 'printf "%s\\n" "$PWD" "$XDG_RUNTIME_DIR" "$DBUS_SESSION_BUS_ADDRESS" "$1"' worker 'argument with spaces'
pwd
''', checkout)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "/", "/run/user/1001", "unix:path=/run/user/1001/bus",
        "argument with spaces", str(checkout),
    ]


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_preflight_precedes_worker_start_and_aborts_on_failure(tmp_path, failure):
    command_log = tmp_path / "commands"
    result = _shell('''
source "$1"
die() { echo "$*" >&2; exit 1; }
id() { echo 1001; }
production_setting() { echo /var/lib/juena-sandbox/workspaces; }
production_worker_installed() { return 0; }
production_worker_image() { return 0; }
production_worker_user() {
    printf '%s\\n' "$*" >> "$COMMAND_LOG"
    case "$1 $2" in
        'podman run') return "$TEST_FAILURE" ;;
        'podman logs') echo 'Sandbox worker ready with 1 slots' ;;
    esac
}
production_worker_start
''', env={**os.environ, "COMMAND_LOG": str(command_log), "TEST_FAILURE": str(int(failure))})
    assert result.returncode == int(failure), result.stderr
    commands = command_log.read_text().splitlines()
    preflight = commands[0]
    assert preflight.startswith("podman run --runtime=crun --rm --pull=never ")
    assert "--userns=keep-id --user=1001:1001 --network=none --read-only" in preflight
    assert "--volume=/var/lib/juena-sandbox/workspaces:/var/lib/juena-sandbox/workspaces:ro" in preflight
    assert "--env-file" not in preflight and "podman.sock" not in preflight
    if failure:
        assert len(commands) == 1
        assert "worker not started" in result.stderr
    else:
        assert commands[1:4] == [
            "systemctl --user enable --now podman.socket",
            "systemctl --user restart podman.service",
            "systemctl --user enable --now juena-sandbox-worker.service",
        ]


@pytest.mark.parametrize("missing", ["", "crun", "newgidmap"])
def test_setup_installs_missing_runtime_even_when_podman_is_present(missing):
    result = _shell('''
source "$1"
die() { echo "$*" >&2; exit 1; }
command() { [ "$1" = -v ] && [ "$2" != "$TEST_MISSING" ]; }
apt-get() { printf '%s\\n' "$*"; }
ensure_production_sandbox_dependencies
''', env={**os.environ, "TEST_MISSING": missing})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (
        ["update", "install -y podman crun uidmap dbus-user-session slirp4netns fuse-overlayfs"]
        if missing else []
    )


def test_generated_worker_settings_select_crun_for_execution_jobs(tmp_path, monkeypatch):
    from juena_core.sandbox.executor import PodmanExecutor, SandboxWorkerSettings

    result = _shell('''
source "$1"
die() { echo "$*" >&2; exit 1; }
production_setting() {
    case "$1" in
        SANDBOX_MEMORY_LIMIT) echo 1g ;;
        SANDBOX_WORKSPACE_LIMIT_BYTES) echo 5368709120 ;;
        SANDBOX_IDLE_TTL_SECONDS) echo 86400 ;;
        *) echo 1 ;;
    esac
}
write_production_worker_limits
''')
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        name, value = line.split("=", 1)
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    settings = SandboxWorkerSettings.from_env()
    executor = PodmanExecutor(settings, WorkspaceStore(tmp_path))
    job = SimpleNamespace(id=uuid4(), workspace_id="test-runtime", timeout_seconds=60,
                          command="python /workspace/.scratch/smoke.py")
    specification = executor._container_kwargs(job)
    assert specification["runtime"] == "crun"
    assert specification["working_dir"] == "/workspace/.scratch"
    assert specification["network_mode"] == "none"
    assert specification["read_only"] is True
    unit = (ROOT / "deploy/juena-sandbox-worker.service").read_text()
    assert f"--runtime={specification['runtime']}" in unit


@pytest.mark.parametrize("proxy_mode", ["standalone", "shared"])
@pytest.mark.parametrize("failure", ["", "deploy", "smoke"])
def test_activation_preserves_secrets_and_rolls_back_on_failure(tmp_path, failure, proxy_mode):
    original = '# keep comment\nSANDBOX_ENABLED=false\nSANDBOX_IDENTITY_SECRET=stable$literal\nPASSWORD=unchanged\n'
    config = tmp_path / "server.env"
    config.write_text(original)
    config.chmod(0o600)
    launcher = tmp_path / "juena"
    launcher.write_text('''#!/bin/sh
[ "$PROD_PROXY_MODE" = "$TEST_PROXY_MODE" ] || exit 2
[ "$TEST_FAILURE" != deploy ]
''')
    launcher.chmod(0o755)
    env = {**os.environ, "TEST_FAILURE": failure, "TEST_PROXY_MODE": proxy_mode}
    result = _shell('''
source "$1"
PROD_ENV_FILE="$2"
REPO_ROOT="$3"
PROD_PROXY_MODE="$TEST_PROXY_MODE"
PROD_EDGE_DIR=/unused
die() { echo "$*" >&2; exit 1; }
production_sandbox_check() { [ "$TEST_FAILURE" != smoke ]; }
production_worker_stop() { echo worker-stopped; }
configure_production_compose() { COMPOSE_PROD=(true); }
enable_production_sandbox
''', config, tmp_path, env=env)
    assert result.returncode == (1 if failure else 0)
    assert config.read_text() == (original if failure else original.replace("SANDBOX_ENABLED=false", "SANDBOX_ENABLED=true"))
    assert config.stat().st_mode & 0o777 == 0o600
    backups = list(tmp_path.glob("server.env.before-sandbox.*"))
    assert len(backups) == 1 and backups[0].read_text() == original
    assert "stable$literal" not in result.stdout + result.stderr
    if failure:
        assert "worker-stopped" in result.stdout


def test_setup_rejects_non_root_before_any_host_changes():
    result = _shell('''
source "$1"
id() { echo 1000; }
die() { echo "$*" >&2; exit 1; }
setup_production_sandbox
''')
    assert result.returncode == 1
    assert "with sudo" in result.stderr


@pytest.mark.parametrize("same_image", [False, True])
def test_worker_imports_the_running_image_only_when_needed(same_image):
    result = _shell('''
source "$1"
COMPOSE_PROD=(compose)
compose() { echo container-id; }
docker() {
    if [ "$1" = inspect ]; then echo sha256:abc
    else echo docker-save >&2; fi
}
production_worker_user() {
    case "$*" in
        'podman image inspect '*) echo "$TEST_IMPORTED" ;;
        'podman load') cat >/dev/null; echo podman-load >&2 ;;
        'podman tag '*) echo "$*" >&2 ;;
    esac
}
production_worker_image
''', env={**os.environ, "TEST_IMPORTED": "abc" if same_image else "old"})
    assert result.returncode == 0, result.stderr
    assert ("docker-save" in result.stderr) is not same_image
    assert ("podman tag sha256:abc localhost/juena-worker:current" in result.stderr) is not same_image


@pytest.mark.parametrize("failure", ["", "initialize", "smoke"])
def test_first_activation_requires_a_passing_smoke_test(failure):
    result = _shell('''
source "$1"
die() { echo "$*" >&2; exit 1; }
production_sandbox_check() {
    if [ "${1:-}" = --initialize ]; then
        echo initialize; [ "$TEST_FAILURE" != initialize ]
    else echo smoke; [ "$TEST_FAILURE" != smoke ]; fi
}
production_worker_start() { echo start; }
production_worker_stop() { echo stop; }
enable_production_sandbox() { echo enable; }
activate_prepared_sandbox
''', env={**os.environ, "TEST_FAILURE": failure})
    assert result.returncode == (1 if failure else 0)
    expected = {"": ["initialize", "start", "smoke", "enable"],
                "initialize": ["initialize"], "smoke": ["initialize", "start", "smoke", "stop"]}
    assert result.stdout.splitlines() == expected[failure]


def test_worker_dsn_changes_only_host_and_port_and_is_not_shell_quoted():
    result = _shell('''
source "$1"
COMPOSE_PROD=(compose)
compose() { shift 4; "$TEST_PYTHON" "$@"; }
production_setting() { echo 15432; }
write_production_worker_dsn
''', env={**os.environ, "TEST_PYTHON": sys.executable,
           "DATABASE_URL": "postgresql://juena:test%40value@postgres:5432/juena?application_name=worker"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DATABASE_URL=postgresql://juena:test%40value@127.0.0.1:15432/juena?application_name=worker\n"


@pytest.mark.parametrize("url", [
    "postgresql://juena:private-value@external.example/juena",
    "postgresql://juena:private'value@postgres/juena",
])
def test_worker_dsn_rejects_unsupported_settings_without_printing_credentials(url):
    result = _shell('''
source "$1"
COMPOSE_PROD=(compose)
compose() { shift 4; "$TEST_PYTHON" "$@"; }
production_setting() { echo 5432; }
write_production_worker_dsn
''', env={**os.environ, "TEST_PYTHON": sys.executable, "DATABASE_URL": url})
    assert result.returncode != 0
    assert result.stdout == ""
    assert "private" not in result.stderr
