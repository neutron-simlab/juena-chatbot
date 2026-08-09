"""Runtime identity resolution and tenant-scoped specialist backends."""

from __future__ import annotations

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

from juena.agents.backends import ReadOnlyFilesystemBackend
from juena.core.config import global_config
from juena.sandbox.backend import PodmanSandboxBackend, ReadOnlySandboxInputsBackend
from juena.sandbox.broker_client import get_broker_client, workspace_id_for


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
        token = global_config.SANDBOX_BROKER_TOKEN
        if not token:
            raise RuntimeError("Sandbox broker token is unavailable")
        return cls(
            user_id=user_id,
            thread_id=thread_id,
            run_id=str(run_id) if run_id else None,
            workspace_id=workspace_id_for(user_id, thread_id, token),
        )


class RuntimePodmanSandboxBackend(BaseSandbox):
    """Resolve the tenant-bound Podman sandbox from the active graph runtime."""

    @staticmethod
    def _resolve() -> PodmanSandboxBackend:
        runtime = get_runtime()
        identity = _SandboxIdentity.from_runtime(runtime)
        return PodmanSandboxBackend(
            broker=get_broker_client(),
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
    token = global_config.SANDBOX_BROKER_TOKEN
    if not token:
        raise RuntimeError("Sandbox broker token is unavailable")
    staged: list[tuple[str, bytes]] = []
    for path, file_data in sorted(files.items()):
        if not path.startswith("/inputs/") or file_data is None:
            continue
        staged.append((path, file_data_to_string(file_data).encode("utf-8")))
    await get_broker_client().stage_inputs(
        workspace_id_for(user_id, thread_id, token), staged
    )


async def delete_runtime_workspace(*, user_id: str, thread_id: str) -> None:
    if not sandbox_enabled():
        return
    token = global_config.SANDBOX_BROKER_TOKEN
    if not token:
        raise RuntimeError("Sandbox broker token is unavailable")
    await get_broker_client().delete_workspace(workspace_id_for(user_id, thread_id, token))
