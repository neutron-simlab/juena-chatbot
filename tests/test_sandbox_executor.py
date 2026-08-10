"""Podman isolation controls and single-shot container lifecycle."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from requests.exceptions import RequestException

from juena.sandbox.executor import (
    ContainerCleanupError,
    JOB_ID_LABEL,
    MANAGED_LABEL,
    PodmanExecutor,
    SandboxSettings,
)
from juena.sandbox.jobs import SandboxJobRecord
from juena.sandbox.workspace import WorkspaceStore


def _executor(tmp_path) -> PodmanExecutor:
    return PodmanExecutor(
        SandboxSettings(
            database_url="postgresql://juena@127.0.0.1:5432/juena",
            podman_uri="unix:///run/user/1000/podman/podman.sock",
            image="juena-sandbox:test",
            concurrency=4,
            cpu_limit="2",
            memory_limit="4g",
            pid_limit=256,
            tmpfs_size="1g",
            runtime=None,
            workspace_root=Path(tmp_path),
            workspace_limit_bytes=5 * 1024**3,
            idle_ttl_seconds=86400,
        ),
        WorkspaceStore(tmp_path),
    )


def _job() -> SandboxJobRecord:
    now = datetime.now(UTC)
    return SandboxJobRecord(
        id=uuid4(),
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        command="python -V",
        timeout_seconds=30,
        max_output_bytes=1000,
        status="running",
        output="",
        exit_code=None,
        truncated=False,
        artifact_names=(),
        created_at=now,
        started_at=now,
        finished_at=None,
    )


@pytest.fixture
def container_kwargs(tmp_path):
    executor = _executor(tmp_path)
    job = _job()
    executor.workspaces.ensure(job.workspace_id, job.tenant_id)
    return executor._container_kwargs(job), job


def _mount(kwargs, target: str) -> dict:
    return next(item for item in kwargs["mounts"] if item["target"] == target)


def test_container_has_dual_timeout_and_ownership_labels(container_kwargs) -> None:
    kwargs, job = container_kwargs
    assert kwargs["command"][:4] == [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "30s",
    ]
    assert kwargs["name"] == f"juena-{job.id.hex}"
    assert kwargs["labels"][MANAGED_LABEL] == "true"
    assert kwargs["labels"][JOB_ID_LABEL] == str(job.id)


def test_container_keeps_isolation_and_resource_limits(container_kwargs) -> None:
    kwargs, _job_record = container_kwargs
    assert kwargs["network_mode"] == "none"
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["no_new_privileges"] is True
    assert kwargs["pids_limit"] == 256
    assert kwargs["mem_limit"] == "4g"
    assert kwargs["cpu_period"] == 100_000
    assert kwargs["cpu_quota"] == 200_000
    assert kwargs["ulimits"] == [
        {"Name": "fsize", "Soft": 5 * 1024**3, "Hard": 5 * 1024**3}
    ]
    assert "security_opt" not in kwargs


def test_container_writes_as_the_worker_so_juena_can_read_artifacts(
    container_kwargs,
) -> None:
    # An explicit uid/gid override would map output files to a subuid neither
    # the worker nor the JüNA container can manage.
    kwargs, _job_record = container_kwargs
    assert kwargs["userns_mode"] == "keep-id"
    assert kwargs["user"] == str(os.getuid())


def test_workspace_is_writable_inputs_are_read_only_and_tmp_is_noexec(
    container_kwargs,
    tmp_path,
) -> None:
    kwargs, job = container_kwargs
    workspace = _mount(kwargs, "/workspace")
    inputs = _mount(kwargs, "/inputs")
    tmp = _mount(kwargs, "/tmp")

    assert workspace["source"] == str(
        (tmp_path / job.workspace_id / "workspace").resolve()
    )
    assert workspace.get("read_only") is not True
    assert inputs["read_only"] is True
    assert tmp["propagation"] == "noexec"


def test_podman_renderer_keeps_every_isolation_control(container_kwargs) -> None:
    from podman.domain.containers_create import CreateMixin

    kwargs, _job_record = container_kwargs
    rendered = CreateMixin._render_payload(dict(kwargs))

    assert rendered["netns"] == {"nsmode": "none"}
    assert rendered["userns"] == {"nsmode": "keep-id"}
    assert rendered["read_only_filesystem"] is True
    assert rendered["cap_drop"] == ["ALL"]
    assert rendered["no_new_privileges"] is True
    assert rendered["resource_limits"]["pids"]["limit"] == 256
    assert rendered["resource_limits"]["memory"]["limit"] == 4 * 1024**3
    assert rendered["resource_limits"]["cpu"]["quota"] == 200_000
    assert rendered["work_dir"] == "/workspace"
    tmp = next(item for item in rendered["mounts"] if item["destination"] == "/tmp")
    assert "noexec" in tmp["options"]


class _Container:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        output: bytes = b"",
        remove_error: Exception | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.output = output
        self.remove_error = remove_error
        self.started = False
        self.removed = False

    def start(self) -> None:
        self.started = True

    def wait(self, *, timeout: int) -> int:
        assert timeout == 40
        return self.exit_code

    def logs(self, **_kwargs):
        yield self.output

    def remove(self, *, force: bool) -> None:
        assert force is True
        if self.remove_error:
            raise self.remove_error
        self.removed = True


def _with_container(executor: PodmanExecutor, container: _Container) -> None:
    executor._client = SimpleNamespace(
        containers=SimpleNamespace(create=lambda **_kwargs: container)
    )


def test_finished_container_is_always_removed(tmp_path) -> None:
    executor = _executor(tmp_path)
    job = _job()
    container = _Container(output=b"3.11.9\n")
    _with_container(executor, container)

    outcome = executor.run(job)

    assert outcome.status == "completed"
    assert outcome.output == "3.11.9\n"
    assert container.started is True
    assert container.removed is True


def test_timeout_exit_code_becomes_timed_out(tmp_path) -> None:
    executor = _executor(tmp_path)
    container = _Container(exit_code=124)
    _with_container(executor, container)

    outcome = executor.run(_job())

    assert outcome.status == "timed_out"
    assert container.removed is True


def test_container_is_removed_even_when_the_run_raises(tmp_path) -> None:
    executor = _executor(tmp_path)
    container = _Container()

    def explode() -> None:
        raise RuntimeError("start failed")

    container.start = explode  # type: ignore[method-assign]
    _with_container(executor, container)

    with pytest.raises(RuntimeError):
        executor.run(_job())

    assert container.removed is True


def test_container_starts_are_serialized_but_execution_overlaps(tmp_path) -> None:
    executor = _executor(tmp_path)
    active_starts = 0
    max_starts = 0
    state_lock = threading.Lock()
    waits = threading.Barrier(2)

    class TrackedContainer(_Container):
        def start(self) -> None:
            nonlocal active_starts, max_starts
            with state_lock:
                active_starts += 1
                max_starts = max(max_starts, active_starts)
            time.sleep(0.05)
            with state_lock:
                active_starts -= 1

        def wait(self, *, timeout: int) -> int:
            assert timeout == 40
            waits.wait(timeout=2)
            return 0

    executor._client = SimpleNamespace(
        containers=SimpleNamespace(create=lambda **_kwargs: TrackedContainer())
    )
    first = _job()
    second = replace(first, id=uuid4(), workspace_id="b" * 64)
    start = threading.Barrier(2)

    def run(job):
        start.wait(timeout=2)
        return executor.run(job)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(run, job) for job in (first, second)]
        assert all(future.result(timeout=5).status == "completed" for future in outcomes)

    assert max_starts == 1


def test_container_cleanup_failure_is_not_suppressed(tmp_path) -> None:
    executor = _executor(tmp_path)
    container = _Container(remove_error=RequestException("remove failed"))
    _with_container(executor, container)

    with pytest.raises(ContainerCleanupError, match="Failed to remove"):
        executor.run(_job())


def test_startup_removes_every_managed_container(tmp_path) -> None:
    executor = _executor(tmp_path)
    containers = [_Container(), _Container()]
    executor._client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda **_kwargs: containers)
    )

    assert executor.remove_all() == 2
    assert all(container.removed for container in containers)


def test_startup_cleanup_failure_is_not_suppressed(tmp_path) -> None:
    executor = _executor(tmp_path)
    container = _Container(remove_error=RequestException("remove failed"))
    executor._client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda **_kwargs: [container])
    )

    with pytest.raises(ContainerCleanupError, match="Failed to remove"):
        executor.remove_all()
