"""Runtime identity resolution and tenant-scoped specialist backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import BackendFactory
from deepagents.backends.utils import file_data_to_string

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
        config = getattr(runtime, "config", {}) or {}
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


class SandboxBackendFactory:
    """Build one composite view for the authenticated runtime invocation."""

    artifacts_root = "/workspace"

    def __init__(
        self,
        *,
        repo_cache_root: Path | None,
        skills_dir: Path | None,
        has_skills: bool,
    ) -> None:
        self._repo_backend = (
            ReadOnlyFilesystemBackend(
                root_dir=repo_cache_root,
                virtual_mode=True,
                label="read-only repository",
            )
            if repo_cache_root is not None
            else None
        )
        self._skills_backend = (
            ReadOnlyFilesystemBackend(
                root_dir=skills_dir,
                virtual_mode=True,
                label="read-only skill",
            )
            if skills_dir is not None and has_skills
            else None
        )

    def __call__(self, runtime: Any) -> CompositeBackend:
        identity = _SandboxIdentity.from_runtime(runtime)
        sandbox = PodmanSandboxBackend(
            broker=get_broker_client(),
            workspace_id=identity.workspace_id,
            user_id=identity.user_id,
            thread_id=identity.thread_id,
            run_id=identity.run_id,
        )
        routes: dict[str, Any] = {
            "/inputs/": ReadOnlySandboxInputsBackend(sandbox),
        }
        if self._repo_backend is not None:
            routes["/repos/"] = self._repo_backend
        if self._skills_backend is not None:
            routes["/skills/"] = self._skills_backend
        return CompositeBackend(
            default=sandbox,
            routes=routes,
            # Deep Agents stores oversized tool output and conversation summaries
            # beneath this root. The sandbox root filesystem is intentionally
            # read-only; `/workspace` is its only writable area.
            artifacts_root="/workspace",
        )


def build_sandbox_backend_factory(
    *,
    repo_cache_root: Path | None,
    skills_dir: Path | None,
    has_skills: bool,
) -> BackendFactory:
    if not sandbox_enabled():
        raise RuntimeError("Cannot build a sandbox backend while SANDBOX_ENABLED is false")
    return SandboxBackendFactory(
        repo_cache_root=repo_cache_root,
        skills_dir=skills_dir,
        has_skills=has_skills,
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
