"""Broker path confinement, Podman policy, and artifact collection."""

from pathlib import Path

import pytest

from juena.sandbox.broker_service import BrokerRuntime, BrokerSettings
from juena.sandbox.schemas import ExecuteRequest


def _runtime(tmp_path) -> BrokerRuntime:
    return BrokerRuntime(
        BrokerSettings(
            token="x" * 32,
            podman_uri="unix:///run/user/1000/podman/podman.sock",
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


@pytest.fixture
def container_kwargs(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.ensure_workspace("a" * 64)
    return runtime._container_kwargs(
        ExecuteRequest(
            workspace_id="a" * 64,
            command="python -V",
            timeout_seconds=30,
            max_output_bytes=1000,
        ),
        "container-name",
    )


def _mount(kwargs, target: str) -> dict:
    return next(item for item in kwargs["mounts"] if item["target"] == target)


def test_container_has_no_network(container_kwargs) -> None:
    assert container_kwargs["network_mode"] == "none"


def test_container_root_filesystem_is_read_only(container_kwargs) -> None:
    assert container_kwargs["read_only"] is True


def test_container_drops_all_capabilities(container_kwargs) -> None:
    assert container_kwargs["cap_drop"] == ["ALL"]


def test_container_forbids_privilege_escalation(container_kwargs) -> None:
    # `security_opt` would silently mean SELinux labels instead.
    assert container_kwargs["no_new_privileges"] is True
    assert "security_opt" not in container_kwargs


def test_container_limits_processes_memory_and_cpu(container_kwargs) -> None:
    assert container_kwargs["pids_limit"] == 256
    assert container_kwargs["mem_limit"] == "4g"
    # 2 CPUs expressed as a fraction of the scheduling period.
    assert container_kwargs["cpu_period"] == 100_000
    assert container_kwargs["cpu_quota"] == 200_000
    assert "cpus" not in container_kwargs


def test_container_caps_single_file_size(container_kwargs) -> None:
    assert container_kwargs["ulimits"] == [
        {"Name": "fsize", "Soft": 5 * 1024**3, "Hard": 5 * 1024**3}
    ]


def test_container_maps_the_sandbox_user_onto_the_broker_account(container_kwargs) -> None:
    # Podman rejects the CLI spelling "keep-id:uid=...,gid=..." over the API.
    assert container_kwargs["userns_mode"] == {
        "nsmode": "keep-id",
        "value": "uid=10001,gid=10001",
    }
    assert "user" not in container_kwargs


def test_workspace_is_writable_and_inputs_are_read_only(container_kwargs, tmp_path) -> None:
    workspace = _mount(container_kwargs, "/workspace")
    inputs = _mount(container_kwargs, "/inputs")

    assert workspace["type"] == "bind"
    assert workspace["relabel"] == "Z"
    assert workspace.get("read_only") is not True
    assert workspace["source"] == str((tmp_path / ("a" * 64) / "workspace").resolve())
    assert inputs["type"] == "bind"
    assert inputs["relabel"] == "Z"
    assert inputs["read_only"] is True


def test_tmpfs_mount_requests_noexec(container_kwargs) -> None:
    """The tmpfs security options must survive podman-py's option renderer.

    Podman applies `nosuid` and `nodev` to API-created tmpfs mounts by default
    but not `noexec`, and podman-py forwards only a fixed set of option keys.
    This asserts on the rendered payload so the protection cannot be dropped
    silently by a podman-py change.
    """
    from podman.domain.containers_create import CreateMixin

    tmpfs = _mount(container_kwargs, "/tmp")
    assert tmpfs["type"] == "tmpfs"
    assert tmpfs["size"] == "1g"

    rendered = CreateMixin._render_payload(dict(container_kwargs))
    options = next(
        item["options"] for item in rendered["mounts"] if item["destination"] == "/tmp"
    )
    assert "noexec" in options
    assert "size=1g" in options


def test_rendered_payload_keeps_every_isolation_control(container_kwargs) -> None:
    """podman-py must not drop a security argument on the floor."""
    from podman.domain.containers_create import CreateMixin

    rendered = CreateMixin._render_payload(dict(container_kwargs))

    assert rendered["netns"] == {"nsmode": "none"}
    assert rendered["userns"] == {"nsmode": "keep-id", "value": "uid=10001,gid=10001"}
    assert rendered["read_only_filesystem"] is True
    assert rendered["cap_drop"] == ["ALL"]
    assert rendered["no_new_privileges"] is True
    assert rendered["resource_limits"]["pids"]["limit"] == 256
    assert rendered["resource_limits"]["memory"]["limit"] == 4 * 1024**3
    assert rendered["resource_limits"]["cpu"]["quota"] == 200_000
    assert rendered["r_limits"] == [
        {"type": "fsize", "hard": 5 * 1024**3, "soft": 5 * 1024**3}
    ]
    assert rendered["work_dir"] == "/workspace"
    inputs = next(item for item in rendered["mounts"] if item["destination"] == "/inputs")
    assert "ro" in inputs["options"]


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
