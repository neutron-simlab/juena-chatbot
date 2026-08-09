"""Broker path confinement, Podman policy, and artifact collection."""

from pathlib import Path

import pytest

from juena.sandbox.broker_service import BrokerRuntime, BrokerSettings
from juena.sandbox.schemas import ExecuteRequest


def _runtime(tmp_path) -> BrokerRuntime:
    return BrokerRuntime(
        BrokerSettings(
            token="x" * 32,
            podman_binary="/usr/bin/podman",
            image="juena-sandbox:test",
            workspace_root=tmp_path,
            concurrency=4,
            cpu_limit="2",
            memory_limit="4g",
            pid_limit=256,
            tmpfs_size="1g",
            workspace_limit_bytes=5 * 1024**3,
            idle_ttl_seconds=86400,
            runtime=None,
        )
    )


def test_path_resolution_rejects_traversal_and_wrong_write_mount(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    workspace.mkdir()
    inputs.mkdir()

    with pytest.raises(ValueError, match="invalid_path"):
        BrokerRuntime._resolve_path(workspace, inputs, "/workspace/../secret", write=True)
    with pytest.raises(PermissionError, match="permission_denied"):
        BrokerRuntime._resolve_path(workspace, inputs, "/inputs/data.csv", write=True)
    with pytest.raises(PermissionError, match="permission_denied"):
        BrokerRuntime._resolve_path(workspace, inputs, "/etc/passwd", write=False)


def test_path_resolution_rejects_symlink_escape(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    outside = tmp_path / "outside"
    workspace.mkdir()
    inputs.mkdir()
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="invalid_path"):
        BrokerRuntime._resolve_path(
            workspace, inputs, "/workspace/escape/secret.txt", write=False
        )


def test_podman_command_contains_isolation_controls(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.ensure_workspace("a" * 64)

    command = runtime._podman_command(
        ExecuteRequest(
            workspace_id="a" * 64,
            command="python -V",
            timeout_seconds=30,
            max_output_bytes=1000,
        ),
        "container-name",
    )

    assert "--network=none" in command
    assert "--userns=keep-id:uid=10001,gid=10001" in command
    assert "--user" not in command
    assert "--read-only" in command
    assert "--cap-drop=all" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pids-limit=256" in command
    assert "--memory=4g" in command
    assert "--cpus=2" in command
    assert "--ulimit=fsize=5368709120:5368709120" in command
    assert "--tmpfs=/tmp:rw,nosuid,nodev,noexec,mode=1777,size=1g" in command
    assert "--env=MPLCONFIGDIR=/tmp/matplotlib" in command
    assert any(item.endswith(":/inputs:ro,Z") for item in command)
    assert any(item.endswith(":/workspace:rw,Z") for item in command)


def test_collects_only_changed_whitelisted_regular_files(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    workspace_id = "a" * 64
    runtime.ensure_workspace(workspace_id)
    output_dir = runtime.workspace_dir(workspace_id) / "outputs"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")
    previous = runtime._artifact_snapshot(workspace_id)
    (output_dir / "new.py").write_text("print('ok')\n", encoding="utf-8")
    (output_dir / "blocked.html").write_text("<script>x</script>", encoding="utf-8")
    (output_dir / "escape.txt").symlink_to(Path("/etc/hosts"))

    artifacts = runtime._collect_artifacts(workspace_id, previous)

    assert [item.filename for item in artifacts] == ["new.py"]
