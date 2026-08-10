"""Rootless-Podman execution core used only by the sandbox worker."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podman import PodmanClient
from podman.errors import APIError, NotFound
from requests.exceptions import RequestException

from juena.sandbox.jobs import SandboxJobRecord
from juena.sandbox.workspace import WorkspaceStore

CPU_PERIOD_MICROSECONDS = 100_000
TMPFS_RAW_OPTION_KEY = "propagation"
PODMAN_ERRORS = (APIError, NotFound, RequestException)
TIMEOUT_EXIT_CODES = {124, 137}
# The worker gives the in-container `timeout` this much grace before it stops
# the container itself.
HOST_TIMEOUT_GRACE_SECONDS = 10

MANAGED_LABEL = "io.juena.sandbox.managed"
JOB_ID_LABEL = "io.juena.sandbox.job-id"


class ContainerCleanupError(RuntimeError):
    """A managed container could not be force-removed."""


def default_podman_uri() -> str:
    return f"unix:///run/user/{os.getuid()}/podman/podman.sock"


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """The worker's whole configuration, read once from its environment."""

    database_url: str
    podman_uri: str
    image: str
    concurrency: int
    cpu_limit: str
    memory_limit: str
    pid_limit: int
    tmpfs_size: str
    runtime: str | None
    workspace_root: Path
    workspace_limit_bytes: int
    idle_ttl_seconds: int

    @classmethod
    def from_env(cls) -> SandboxSettings:
        url = os.getenv("DATABASE_URL", "")
        if not url:
            raise RuntimeError("DATABASE_URL is required")
        return cls(
            database_url=url,
            podman_uri=os.getenv("SANDBOX_PODMAN_URI") or default_podman_uri(),
            image=os.getenv("SANDBOX_IMAGE", "juena-sandbox:latest"),
            concurrency=int(os.getenv("SANDBOX_CONCURRENCY", "4")),
            cpu_limit=os.getenv("SANDBOX_CPU_LIMIT", "2"),
            memory_limit=os.getenv("SANDBOX_MEMORY_LIMIT", "4g"),
            pid_limit=int(os.getenv("SANDBOX_PID_LIMIT", "256")),
            tmpfs_size=os.getenv("SANDBOX_TMPFS_SIZE", "1g"),
            runtime=os.getenv("SANDBOX_OCI_RUNTIME") or None,
            workspace_root=Path(
                os.getenv("SANDBOX_WORKSPACE_ROOT", "/var/lib/juena-sandbox/workspaces")
            ),
            workspace_limit_bytes=int(
                os.getenv("SANDBOX_WORKSPACE_LIMIT_BYTES", str(5 * 1024**3))
            ),
            idle_ttl_seconds=int(os.getenv("SANDBOX_IDLE_TTL_SECONDS", str(24 * 3600))),
        )


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: str
    output: str
    exit_code: int | None
    truncated: bool
    artifact_names: list[str]


class PodmanExecutor:
    def __init__(
        self,
        settings: SandboxSettings,
        workspaces: WorkspaceStore,
    ) -> None:
        self.settings = settings
        self.workspaces = workspaces
        self._client: PodmanClient | None = None
        self._start_lock = threading.Lock()

    @property
    def client(self) -> PodmanClient:
        if self._client is None:
            raise RuntimeError("Sandbox worker is not connected to Podman")
        return self._client

    def connect(self) -> None:
        self._client = PodmanClient(
            base_url=self.settings.podman_uri,
            max_pool_size=max(self.settings.concurrency, 10),
        )

    def close(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                self._client.close()
            self._client = None

    def validate_startup(self) -> None:
        if os.geteuid() == 0:
            raise RuntimeError("Sandbox worker refuses to run as root")
        if self.settings.concurrency < 1:
            raise RuntimeError("SANDBOX_CONCURRENCY must be at least 1")
        try:
            reachable = self.client.ping()
        except (APIError, RequestException) as exc:
            raise RuntimeError(
                f"Podman socket is unavailable at {self.settings.podman_uri}: {exc}"
            ) from exc
        if not reachable:
            raise RuntimeError(f"Podman socket is unavailable at {self.settings.podman_uri}")
        if not self.client.images.exists(self.settings.image):
            raise RuntimeError(f"Sandbox image is unavailable: {self.settings.image}")

    def _container_kwargs(self, job: SandboxJobRecord) -> dict[str, Any]:
        workspace = self.workspaces.workspace_dir(job.workspace_id).resolve()
        inputs = self.workspaces.inputs_dir(job.workspace_id).resolve()
        limit = self.workspaces.limit_bytes
        kwargs: dict[str, Any] = {
            "image": self.settings.image,
            "command": [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{job.timeout_seconds}s",
                "/bin/sh",
                "-lc",
                job.command,
            ],
            "name": f"juena-{job.id.hex}",
            "labels": {MANAGED_LABEL: "true", JOB_ID_LABEL: str(job.id)},
            # Keep workspace files owned by the host worker and readable by JüNA.
            "userns_mode": "keep-id",
            "user": str(os.getuid()),
            "network_mode": "none",
            "read_only": True,
            "cap_drop": ["ALL"],
            "no_new_privileges": True,
            "pids_limit": self.settings.pid_limit,
            "mem_limit": self.settings.memory_limit,
            "cpu_period": CPU_PERIOD_MICROSECONDS,
            "cpu_quota": int(float(self.settings.cpu_limit) * CPU_PERIOD_MICROSECONDS),
            "ulimits": [{"Name": "fsize", "Soft": limit, "Hard": limit}],
            "working_dir": "/workspace",
            "environment": {
                "HOME": "/tmp",
                "MPLCONFIGDIR": "/tmp/matplotlib",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "mounts": [
                {
                    "type": "tmpfs",
                    "source": "tmpfs",
                    "target": "/tmp",
                    "size": self.settings.tmpfs_size,
                    "mode": "1777",
                    TMPFS_RAW_OPTION_KEY: "noexec",
                },
                {
                    "type": "bind",
                    "source": str(workspace),
                    "target": "/workspace",
                    "relabel": "Z",
                },
                {
                    "type": "bind",
                    "source": str(inputs),
                    "target": "/inputs",
                    "read_only": True,
                    "relabel": "Z",
                },
            ],
        }
        if self.settings.runtime:
            kwargs["runtime"] = self.settings.runtime
        return kwargs

    @staticmethod
    def _wait_for_exit(container: Any, timeout: int) -> tuple[int | None, bool]:
        try:
            return container.wait(timeout=timeout), False
        except (APIError, RequestException):
            pass
        with suppress(*PODMAN_ERRORS):
            container.reload()
            state = container.attrs.get("State", {})
            if not state.get("Running", False):
                exit_code = state.get("ExitCode")
                return (exit_code if isinstance(exit_code, int) else None), False
        return None, True

    @staticmethod
    def _read_logs(container: Any, max_bytes: int) -> tuple[bytes, bool]:
        buffer = bytearray()
        truncated = False
        frames: Iterator[bytes] = container.logs(
            stdout=True,
            stderr=True,
            stream=True,
            follow=False,
        )
        with closing(frames):
            for chunk in frames:
                remaining = max_bytes - len(buffer)
                if len(chunk) >= remaining:
                    buffer.extend(chunk[:remaining])
                    truncated = len(chunk) > remaining
                    break
                buffer.extend(chunk)
        return bytes(buffer), truncated

    def run(self, job: SandboxJobRecord) -> ExecutionOutcome:
        """Run one claimed job to completion. The container never outlives it."""

        self.workspaces.ensure(job.workspace_id, job.tenant_id)
        self.workspaces.enforce_limit(job.workspace_id)
        self.workspaces.ensure_outputs(job.workspace_id)
        started_at = job.started_at or job.created_at

        with self._start_lock:
            container = self.client.containers.create(**self._container_kwargs(job))
            try:
                container.start()
            except Exception:
                self._remove(container)
                raise
        try:
            exit_code, host_timed_out = self._wait_for_exit(
                container,
                job.timeout_seconds + HOST_TIMEOUT_GRACE_SECONDS,
            )
            if host_timed_out:
                with suppress(*PODMAN_ERRORS):
                    container.stop(timeout=0)
            output, truncated = self._read_logs(container, job.max_output_bytes)
        finally:
            self._remove(container)

        self.workspaces.enforce_limit(job.workspace_id)
        timed_out = host_timed_out or exit_code in TIMEOUT_EXIT_CODES
        return ExecutionOutcome(
            "timed_out" if timed_out else "failed" if exit_code != 0 else "completed",
            output.decode("utf-8", errors="replace"),
            exit_code,
            truncated,
            self.workspaces.artifacts_since(job.workspace_id, started_at),
        )

    @staticmethod
    def _remove(container: Any) -> None:
        try:
            container.remove(force=True)
        except NotFound:
            return
        except (APIError, RequestException) as exc:
            raise ContainerCleanupError("Failed to remove managed sandbox container") from exc

    def remove_all(self) -> int:
        """Force-remove every managed container. Only one worker ever runs."""

        removed = 0
        for container in self.client.containers.list(
            all=True,
            filters={"label": f"{MANAGED_LABEL}=true"},
        ):
            self._remove(container)
            removed += 1
        return removed
