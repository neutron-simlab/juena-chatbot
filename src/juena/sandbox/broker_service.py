"""Rootless-Podman broker service for untrusted scientific commands.

Run this service as a dedicated, unprivileged host account. The JüNA API talks
to it over a private network and never receives a Podman or Docker socket.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import os
import shutil
import subprocess
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import monotonic, sleep, time
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status

from juena.sandbox.constants import BROKER_AUTH_HEADER
from juena.sandbox.schemas import (
    BrokerArtifact,
    BrokerFileResult,
    DownloadFilesRequest,
    ExecuteRequest,
    ExecuteResult,
    StageInputsRequest,
    UploadFilesRequest,
)

MAX_STAGE_BYTES = 100 * 1024 * 1024
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_FILE_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_ARTIFACTS = 4
MAX_FILES = 8
COLLECTED_EXTENSIONS = frozenset(
    {
        ".csv",
        ".ipynb",
        ".json",
        ".log",
        ".md",
        ".npy",
        ".npz",
        ".pdf",
        ".png",
        ".py",
        ".sh",
        ".tsv",
        ".txt",
    }
)


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    token: str
    podman_binary: str
    image: str
    workspace_root: Path
    concurrency: int
    cpu_limit: str
    memory_limit: str
    pid_limit: int
    tmpfs_size: str
    workspace_limit_bytes: int
    idle_ttl_seconds: int
    runtime: str | None

    @classmethod
    def from_env(cls) -> "BrokerSettings":
        return cls(
            token=os.getenv("SANDBOX_BROKER_TOKEN", ""),
            podman_binary=os.getenv("SANDBOX_PODMAN_BINARY", "/usr/bin/podman"),
            image=os.getenv("SANDBOX_IMAGE", "juena-sandbox:latest"),
            workspace_root=Path(
                os.getenv("SANDBOX_WORKSPACE_ROOT", "/var/lib/juena-sandbox/workspaces")
            ),
            concurrency=int(os.getenv("SANDBOX_CONCURRENCY", "4")),
            cpu_limit=os.getenv("SANDBOX_CPU_LIMIT", "2"),
            memory_limit=os.getenv("SANDBOX_MEMORY_LIMIT", "4g"),
            pid_limit=int(os.getenv("SANDBOX_PID_LIMIT", "256")),
            tmpfs_size=os.getenv("SANDBOX_TMPFS_SIZE", "1g"),
            workspace_limit_bytes=int(
                os.getenv("SANDBOX_WORKSPACE_LIMIT_BYTES", str(5 * 1024**3))
            ),
            idle_ttl_seconds=int(os.getenv("SANDBOX_IDLE_TTL_SECONDS", str(24 * 3600))),
            runtime=os.getenv("SANDBOX_OCI_RUNTIME") or None,
        )


class BrokerRuntime:
    def __init__(self, settings: BrokerSettings) -> None:
        self.settings = settings
        self.slots = threading.BoundedSemaphore(settings.concurrency)
        self._locks_guard = threading.Lock()
        self._workspace_locks: dict[str, threading.RLock] = {}

    def validate_startup(self) -> None:
        if os.geteuid() == 0:
            raise RuntimeError("Sandbox broker refuses to run as root")
        if len(self.settings.token) < 32:
            raise RuntimeError("SANDBOX_BROKER_TOKEN must contain at least 32 characters")
        binary = Path(self.settings.podman_binary)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Container engine binary is unavailable: {binary}")
        self.settings.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        probe = subprocess.run(
            [self.settings.podman_binary, "image", "exists", self.settings.image],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"Sandbox image is unavailable: {self.settings.image}")

    def lock_for(self, workspace_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._workspace_locks.setdefault(workspace_id, threading.RLock())

    def workspace_root(self, workspace_id: str) -> Path:
        return self.settings.workspace_root / workspace_id

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "workspace"

    def inputs_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "inputs"

    def ensure_workspace(self, workspace_id: str) -> None:
        root = self.workspace_root(workspace_id)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workspace_dir(workspace_id).mkdir(exist_ok=True, mode=0o700)
        self.inputs_dir(workspace_id).mkdir(exist_ok=True, mode=0o700)
        os.utime(root, None)

    @staticmethod
    def _directory_size(root: Path) -> int:
        return sum(
            path.stat().st_size
            for path in root.rglob("*")
            if not path.is_symlink() and path.is_file()
        )

    def enforce_workspace_limit(self, workspace_id: str) -> None:
        if (
            self._directory_size(self.workspace_dir(workspace_id))
            > self.settings.workspace_limit_bytes
        ):
            raise RuntimeError("Sandbox workspace exceeds its 5 GB limit")

    @staticmethod
    def _resolve_path(workspace: Path, inputs: Path, path: str, *, write: bool) -> Path:
        posix = PurePosixPath(path)
        if not path.startswith("/") or ".." in posix.parts or "~" in posix.parts:
            raise ValueError("invalid_path")
        if path == "/workspace" or path.startswith("/workspace/"):
            relative = posix.relative_to("/workspace")
            root = workspace
        elif not write and (path == "/inputs" or path.startswith("/inputs/")):
            relative = posix.relative_to("/inputs")
            root = inputs
        else:
            raise PermissionError("permission_denied")
        resolved = (root / Path(*relative.parts)).resolve()
        if resolved != root.resolve() and root.resolve() not in resolved.parents:
            raise ValueError("invalid_path")
        return resolved

    def stage_inputs(self, request: StageInputsRequest) -> None:
        total = 0
        decoded: list[tuple[str, bytes]] = []
        for item in request.files:
            if not item.path.startswith("/inputs/"):
                raise ValueError("Only /inputs files may be staged")
            content = base64.b64decode(item.content_base64, validate=True)
            total += len(content)
            if total > MAX_STAGE_BYTES:
                raise ValueError("Staged inputs exceed the 100 MB limit")
            decoded.append((item.path, content))

        with self.lock_for(request.workspace_id):
            self.ensure_workspace(request.workspace_id)
            inputs = self.inputs_dir(request.workspace_id)
            replacement = inputs.with_name(f"inputs-{uuid4().hex}.tmp")
            replacement.mkdir(mode=0o700)
            try:
                for path, content in decoded:
                    target = self._resolve_path(
                        self.workspace_dir(request.workspace_id), replacement, path, write=False
                    )
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target.write_bytes(content)
                old = inputs.with_name(f"inputs-{uuid4().hex}.old")
                os.replace(inputs, old)
                os.replace(replacement, inputs)
                shutil.rmtree(old, ignore_errors=True)
            finally:
                shutil.rmtree(replacement, ignore_errors=True)

    def upload_files(self, request: UploadFilesRequest) -> list[BrokerFileResult]:
        results: list[BrokerFileResult] = []
        total = 0
        with self.lock_for(request.workspace_id):
            self.ensure_workspace(request.workspace_id)
            for item in request.files:
                try:
                    content = base64.b64decode(item.content_base64, validate=True)
                    total += len(content)
                    if total > MAX_STAGE_BYTES:
                        raise ValueError("Uploaded files exceed the 100 MB transfer limit")
                    target = self._resolve_path(
                        self.workspace_dir(request.workspace_id),
                        self.inputs_dir(request.workspace_id),
                        item.path,
                        write=True,
                    )
                    if target.exists():
                        results.append(
                            BrokerFileResult(path=item.path, error="file already exists")
                        )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target.write_bytes(content)
                    results.append(BrokerFileResult(path=item.path))
                except (ValueError, PermissionError) as exc:
                    results.append(BrokerFileResult(path=item.path, error=str(exc)))
            self.enforce_workspace_limit(request.workspace_id)
        return results

    def download_files(self, request: DownloadFilesRequest) -> list[BrokerFileResult]:
        results: list[BrokerFileResult] = []
        total = 0
        with self.lock_for(request.workspace_id):
            self.ensure_workspace(request.workspace_id)
            for path in request.paths:
                try:
                    target = self._resolve_path(
                        self.workspace_dir(request.workspace_id),
                        self.inputs_dir(request.workspace_id),
                        path,
                        write=False,
                    )
                    if not target.is_file():
                        results.append(BrokerFileResult(path=path, error="file_not_found"))
                        continue
                    size = target.stat().st_size
                    if total + size > MAX_STAGE_BYTES:
                        results.append(
                            BrokerFileResult(path=path, error="download transfer limit exceeded")
                        )
                        continue
                    total += size
                    content = target.read_bytes()
                    results.append(
                        BrokerFileResult(
                            path=path,
                            content_base64=base64.b64encode(content).decode("ascii"),
                        )
                    )
                except (OSError, ValueError, PermissionError) as exc:
                    results.append(BrokerFileResult(path=path, error=str(exc)))
        return results

    def _podman_command(self, request: ExecuteRequest, container_name: str) -> list[str]:
        settings = self.settings
        workspace = self.workspace_dir(request.workspace_id).resolve()
        inputs = self.inputs_dir(request.workspace_id).resolve()
        command = [
            settings.podman_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network=none",
            "--read-only",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            f"--pids-limit={settings.pid_limit}",
            f"--memory={settings.memory_limit}",
            f"--cpus={settings.cpu_limit}",
            f"--ulimit=fsize={settings.workspace_limit_bytes}:{settings.workspace_limit_bytes}",
            f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,mode=1777,size={settings.tmpfs_size}",
            f"--volume={workspace}:/workspace:rw,Z",
            f"--volume={inputs}:/inputs:ro,Z",
            "--workdir=/workspace",
            "--env=HOME=/tmp",
            "--env=MPLCONFIGDIR=/tmp/matplotlib",
            "--env=PYTHONDONTWRITEBYTECODE=1",
        ]
        # The image runs as the fixed `sandbox` user (10001:10001). Map that
        # identity to the unprivileged broker account so bind-mounted workspace
        # files stay writable without probing host or VM UID/GID values.
        command[6:6] = ["--userns=keep-id:uid=10001,gid=10001"]
        if settings.runtime:
            command.append(f"--runtime={settings.runtime}")
        command.extend([settings.image, "/bin/sh", "-lc", request.command])
        return command

    @staticmethod
    def _drain_output(stream, buffer: bytearray, max_bytes: int, truncated: list[bool]) -> None:
        while chunk := stream.read(8192):
            remaining = max_bytes - len(buffer)
            if remaining > 0:
                buffer.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    def _artifact_snapshot(self, workspace_id: str) -> dict[str, tuple[int, int]]:
        output_dir = self.workspace_dir(workspace_id) / "outputs"
        if output_dir.is_symlink() or not output_dir.is_dir():
            return {}
        return {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in output_dir.iterdir()
            if not path.is_symlink() and path.is_file()
        }

    def _collect_artifacts(
        self,
        workspace_id: str,
        previous: dict[str, tuple[int, int]],
    ) -> list[BrokerArtifact]:
        output_dir = self.workspace_dir(workspace_id) / "outputs"
        artifacts: list[BrokerArtifact] = []
        if output_dir.is_symlink() or not output_dir.is_dir():
            return artifacts
        output_root = output_dir.resolve()
        image_count = 0
        for path in sorted(output_dir.iterdir()):
            if len(artifacts) >= MAX_FILES:
                break
            if path.is_symlink() or not path.is_file():
                continue
            extension = path.suffix.lower()
            if extension not in COLLECTED_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved.parent != output_root:
                continue
            stat = path.stat()
            if previous.get(path.name) == (stat.st_mtime_ns, stat.st_size):
                continue
            if extension == ".png":
                if image_count >= MAX_ARTIFACTS or stat.st_size > MAX_ARTIFACT_BYTES:
                    continue
                image_count += 1
            elif stat.st_size > MAX_FILE_ARTIFACT_BYTES:
                continue
            artifacts.append(
                BrokerArtifact(
                    filename=path.name,
                    content_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                )
            )
        return artifacts

    def execute(self, request: ExecuteRequest) -> ExecuteResult:
        deadline = monotonic() + request.timeout_seconds
        if not self.slots.acquire(timeout=request.timeout_seconds):
            return ExecuteResult(
                job_id=str(uuid4()),
                output="Sandbox queue wait timed out",
                exit_code=None,
                status="timed_out",
            )
        job_id = str(uuid4())
        container_name = f"juena-{request.workspace_id[:12]}-{job_id[:8]}"
        try:
            with self.lock_for(request.workspace_id):
                self.ensure_workspace(request.workspace_id)
                self.enforce_workspace_limit(request.workspace_id)
                outputs = self.workspace_dir(request.workspace_id) / "outputs"
                if outputs.is_symlink() or (outputs.exists() and not outputs.is_dir()):
                    outputs.unlink(missing_ok=True)
                outputs.mkdir(exist_ok=True, mode=0o700)
                previous_artifacts = self._artifact_snapshot(request.workspace_id)

                process = subprocess.Popen(
                    self._podman_command(request, container_name),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                assert process.stdout is not None
                output = bytearray()
                truncated = [False]
                reader = threading.Thread(
                    target=self._drain_output,
                    args=(process.stdout, output, request.max_output_bytes, truncated),
                    daemon=True,
                )
                reader.start()
                timed_out = False
                workspace_exceeded = False
                while process.poll() is None:
                    if monotonic() >= deadline:
                        timed_out = True
                        break
                    if (
                        self._directory_size(self.workspace_dir(request.workspace_id))
                        > self.settings.workspace_limit_bytes
                    ):
                        workspace_exceeded = True
                        break
                    sleep(0.25)
                if timed_out or workspace_exceeded:
                    process.kill()
                    subprocess.run(
                        [self.settings.podman_binary, "rm", "-f", container_name],
                        check=False,
                        capture_output=True,
                        timeout=30,
                    )
                    exit_code = None
                else:
                    exit_code = process.returncode
                reader.join(timeout=10)
                if workspace_exceeded:
                    message = b"\nSandbox workspace limit exceeded\n"
                    remaining = request.max_output_bytes - len(output)
                    if remaining > 0:
                        output.extend(message[:remaining])
                    truncated[0] = truncated[0] or len(message) > remaining
                else:
                    self.enforce_workspace_limit(request.workspace_id)
                return ExecuteResult(
                    job_id=job_id,
                    output=output.decode("utf-8", errors="replace"),
                    exit_code=exit_code,
                    truncated=truncated[0],
                    status=(
                        "timed_out"
                        if timed_out
                        else "failed"
                        if workspace_exceeded or exit_code != 0
                        else "completed"
                    ),
                    artifacts=self._collect_artifacts(request.workspace_id, previous_artifacts),
                )
        except Exception as exc:
            return ExecuteResult(
                job_id=job_id,
                output=f"Sandbox job failed: {exc}",
                exit_code=None,
                status="failed",
            )
        finally:
            self.slots.release()

    def delete_workspace(self, workspace_id: str) -> None:
        with self.lock_for(workspace_id):
            shutil.rmtree(self.workspace_root(workspace_id), ignore_errors=True)
        with self._locks_guard:
            self._workspace_locks.pop(workspace_id, None)

    def cleanup_expired(self) -> int:
        cutoff = time() - self.settings.idle_ttl_seconds
        removed = 0
        root = self.settings.workspace_root
        if not root.is_dir():
            return removed
        for candidate in root.iterdir():
            if not candidate.is_dir() or candidate.stat().st_mtime >= cutoff:
                continue
            self.delete_workspace(candidate.name)
            removed += 1
        return removed


settings = BrokerSettings.from_env()
runtime = BrokerRuntime(settings)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(3600)
        await asyncio.to_thread(runtime.cleanup_expired)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    runtime.validate_startup()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def authorize(
    token: str | None = Header(default=None, alias=BROKER_AUTH_HEADER),
) -> None:
    if token is None or not hmac.compare_digest(token, settings.token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/health", dependencies=[Depends(authorize)])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/inputs/stage", dependencies=[Depends(authorize)])
def stage_inputs(request: StageInputsRequest) -> dict[str, str]:
    try:
        runtime.stage_inputs(request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "staged"}


@app.post("/v1/files/upload", dependencies=[Depends(authorize)])
def upload_files(request: UploadFilesRequest) -> dict[str, object]:
    return {"files": [item.model_dump(mode="json") for item in runtime.upload_files(request)]}


@app.post("/v1/files/download", dependencies=[Depends(authorize)])
def download_files(request: DownloadFilesRequest) -> dict[str, object]:
    return {"files": [item.model_dump(mode="json") for item in runtime.download_files(request)]}


@app.post("/v1/execute", dependencies=[Depends(authorize)])
def execute(request: ExecuteRequest) -> ExecuteResult:
    return runtime.execute(request)


@app.delete("/v1/workspaces/{workspace_id}", dependencies=[Depends(authorize)])
def delete_workspace(workspace_id: str) -> dict[str, str]:
    if len(workspace_id) != 64 or any(char not in "0123456789abcdef" for char in workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace id")
    runtime.delete_workspace(workspace_id)
    return {"status": "deleted"}
