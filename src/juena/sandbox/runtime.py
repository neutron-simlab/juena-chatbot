"""Runtime identity resolution and tenant-scoped specialist backends."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.utils import file_data_to_string
from langgraph.config import get_config
from langgraph.runtime import get_runtime

from juena.agents.backends import (
    FINDINGS_PREFIX,
    FindingsStateBackend,
    ReadOnlyFilesystemBackend,
)
from juena.core.config import global_config
from juena.sandbox.backend import PodmanSandboxBackend, ReadOnlySandboxInputsBackend
from juena.sandbox.jobs import SandboxJobs, tenant_id_for, workspace_id_for
from juena.sandbox.workspace import WorkspaceStore


def sandbox_enabled() -> bool:
    return bool(global_config.SANDBOX_ENABLED)


def _context_value(context: Any, name: str) -> str | None:
    if isinstance(context, dict):
        value = context.get(name)
    else:
        value = getattr(context, name, None)
    return str(value) if value else None


@dataclass(frozen=True, slots=True)
class _SandboxIdentity:
    user_id: str
    thread_id: str
    run_id: str | None
    tenant_id: str
    workspace_id: str

    @classmethod
    def from_runtime(cls, runtime: Any) -> "_SandboxIdentity":
        context = getattr(runtime, "context", None)
        user_id = _context_value(context, "user_id")
        thread_id = _context_value(context, "thread_id")
        if not user_id or not thread_id:
            raise RuntimeError("Sandbox execution requires authenticated user_id and thread_id")
        config = getattr(runtime, "config", None)
        if config is None:
            try:
                config = get_config()
            except (KeyError, RuntimeError):
                config = {}
        run_id = config.get("run_id") if hasattr(config, "get") else None
        secret = global_config.SANDBOX_IDENTITY_SECRET
        if not secret:
            raise RuntimeError("Sandbox identity secret is unavailable")
        return cls(
            user_id=user_id,
            thread_id=thread_id,
            run_id=str(run_id) if run_id else None,
            tenant_id=tenant_id_for(user_id, secret),
            workspace_id=workspace_id_for(user_id, thread_id, secret),
        )


_workspace_store: WorkspaceStore | None = None


def get_workspace_store() -> WorkspaceStore:
    global _workspace_store
    if _workspace_store is None or _workspace_store.root != global_config.SANDBOX_WORKSPACE_ROOT:
        _workspace_store = WorkspaceStore(global_config.SANDBOX_WORKSPACE_ROOT)
    return _workspace_store


def set_workspace_store_for_tests(store: WorkspaceStore | None) -> None:
    global _workspace_store
    _workspace_store = store


# The application's shared connection pool. It lives here rather than in
# `jobs.py` so that the standalone worker, which imports `jobs.py`, never pulls
# in — or has to satisfy — the application's configuration.
_jobs: SandboxJobs | None = None
_jobs_lock = threading.Lock()


def get_sandbox_jobs() -> SandboxJobs:
    global _jobs
    if not global_config.SANDBOX_ENABLED:
        raise RuntimeError("Sandbox execution is disabled")
    if not global_config.DATABASE_URL:
        raise RuntimeError("Sandbox jobs need DATABASE_URL")
    if _jobs is None:
        with _jobs_lock:
            if _jobs is None:
                jobs = SandboxJobs(
                    global_config.DATABASE_URL,
                    concurrency=global_config.SANDBOX_CONCURRENCY,
                )
                jobs.open()
                _jobs = jobs
    return _jobs


def set_sandbox_jobs_for_tests(jobs: SandboxJobs | None) -> None:
    global _jobs
    _jobs = jobs


def close_sandbox_jobs() -> None:
    global _jobs
    if _jobs is not None:
        _jobs.close()
        _jobs = None


class RuntimePodmanSandboxBackend(BaseSandbox):
    """Resolve the tenant-bound Podman sandbox from the active graph runtime."""

    @staticmethod
    def _resolve() -> PodmanSandboxBackend:
        runtime = get_runtime()
        identity = _SandboxIdentity.from_runtime(runtime)
        return PodmanSandboxBackend(
            jobs=get_sandbox_jobs(),
            workspaces=get_workspace_store(),
            tenant_id=identity.tenant_id,
            workspace_id=identity.workspace_id,
            user_id=identity.user_id,
            thread_id=identity.thread_id,
            run_id=identity.run_id,
        )

    @property
    def id(self) -> str:
        return self._resolve().id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._resolve().execute(command, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._resolve().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._resolve().download_files(paths)


def build_sandbox_backend(
    *,
    repo_cache_root: Path | None,
    skills_dir: Path | None,
    has_skills: bool,
) -> CompositeBackend:
    if not sandbox_enabled():
        raise RuntimeError("Cannot build a sandbox backend while SANDBOX_ENABLED is false")

    sandbox = RuntimePodmanSandboxBackend()
    routes: dict[str, Any] = {
        "/inputs/": ReadOnlySandboxInputsBackend(sandbox),
        # Graph state, never the container. The sandbox root is read-only, so a
        # specialist writing here got `Errno 30: Read-only file system`; and even
        # a writable container would hide findings from a background specialist,
        # which runs with no sandbox at all.
        FINDINGS_PREFIX: FindingsStateBackend(),
    }
    if repo_cache_root is not None:
        routes["/repos/"] = ReadOnlyFilesystemBackend(
            root_dir=repo_cache_root,
            virtual_mode=True,
            label="read-only repository",
        )
    if skills_dir is not None and has_skills:
        routes["/skills/"] = ReadOnlyFilesystemBackend(
            root_dir=skills_dir,
            virtual_mode=True,
            label="read-only skill",
        )
    return CompositeBackend(
        default=sandbox,
        routes=routes,
        # Deep Agents stores oversized tool output and conversation summaries
        # beneath this root. The sandbox root filesystem is intentionally
        # read-only; `/workspace` is its only writable area.
        artifacts_root="/workspace",
    )


async def stage_runtime_inputs(
    *,
    user_id: str,
    thread_id: str,
    files: dict[str, Any],
) -> None:
    if not sandbox_enabled():
        return
    secret = global_config.SANDBOX_IDENTITY_SECRET
    if not secret:
        raise RuntimeError("Sandbox identity secret is unavailable")
    tenant_id = tenant_id_for(user_id, secret)
    workspace_id = workspace_id_for(user_id, thread_id, secret)
    staged: list[tuple[str, bytes]] = []
    for path, file_data in sorted(files.items()):
        if not path.startswith("/inputs/") or file_data is None:
            continue
        staged.append((path, file_data_to_string(file_data).encode("utf-8")))
    jobs = get_sandbox_jobs()

    def stage() -> None:
        with jobs.workspace_guard(workspace_id) as available:
            if not available:
                raise RuntimeError("Sandbox workspace is busy")
            get_workspace_store().stage_inputs(workspace_id, tenant_id, staged)

    await asyncio.to_thread(stage)


async def delete_runtime_workspace(*, user_id: str, thread_id: str) -> None:
    if not sandbox_enabled():
        return
    secret = global_config.SANDBOX_IDENTITY_SECRET
    if not secret:
        raise RuntimeError("Sandbox identity secret is unavailable")
    tenant_id = tenant_id_for(user_id, secret)
    workspace_id = workspace_id_for(user_id, thread_id, secret)
    jobs = get_sandbox_jobs()

    def delete() -> None:
        with jobs.workspace_guard(workspace_id) as available:
            if available:
                get_workspace_store().delete(workspace_id, tenant_id)

    await asyncio.to_thread(delete)
