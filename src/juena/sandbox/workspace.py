"""Shared sandbox workspace with tenant ownership and path confinement."""

from __future__ import annotations

import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from time import time
from uuid import uuid4

from juena.sandbox.constants import (
    COLLECTED_EXTENSIONS,
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACTS_PER_EXECUTION,
    MAX_FILE_ARTIFACT_BYTES,
    MAX_FILE_ARTIFACTS_PER_EXECUTION,
)

MAX_TRANSFER_BYTES = 100 * 1024 * 1024
OWNER_FILE = ".owner"
DIRECTORY_MODE = 0o2770
FILE_MODE = 0o660


@dataclass(frozen=True, slots=True)
class WorkspaceFileResult:
    path: str
    content: bytes | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactScan:
    """Accepted outputs plus a bounded account of files the worker refused."""

    accepted: tuple[str, ...] = ()
    dropped: tuple[tuple[str, str], ...] = ()
    omitted_drops: int = 0


class WorkspaceOwnershipError(RuntimeError):
    pass


def assert_safe_path(path: str, message: str = "invalid_path") -> PurePosixPath:
    """Reject anything that is not a plain absolute path inside the sandbox."""

    posix = PurePosixPath(path)
    if not path.startswith("/") or ".." in posix.parts or "~" in posix.parts:
        raise ValueError(message)
    return posix


class WorkspaceStore:
    def __init__(
        self,
        root: Path,
        *,
        limit_bytes: int = 5 * 1024**3,
        idle_ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.root = root
        self.limit_bytes = limit_bytes
        self.idle_ttl_seconds = idle_ttl_seconds

    def workspace_root(self, workspace_id: str) -> Path:
        return self.root / workspace_id

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "workspace"

    def inputs_dir(self, workspace_id: str) -> Path:
        return self.workspace_root(workspace_id) / "inputs"

    @staticmethod
    def _make_dir(path: Path) -> Path:
        """Create a setgid directory independent of the caller's umask."""
        path.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            path.chmod(DIRECTORY_MODE)
        return path

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.write_bytes(content)
        with suppress(OSError):
            path.chmod(FILE_MODE)

    def ensure(self, workspace_id: str, tenant_id: str) -> None:
        root = self.workspace_root(workspace_id)
        if root.is_symlink():
            raise WorkspaceOwnershipError("Sandbox workspace root cannot be a symlink")
        self._make_dir(root)
        owner = root / OWNER_FILE
        if owner.is_symlink():
            raise WorkspaceOwnershipError("Sandbox workspace owner marker is invalid")
        if owner.exists():
            if owner.read_text(encoding="ascii").strip() != tenant_id:
                raise WorkspaceOwnershipError("Sandbox workspace belongs to another tenant")
        else:
            self._write(owner, f"{tenant_id}\n".encode("ascii"))
        self._make_dir(self.workspace_dir(workspace_id))
        self._make_dir(self.inputs_dir(workspace_id))
        os.utime(root, None)

    def ensure_outputs(self, workspace_id: str) -> Path:
        outputs = self.workspace_dir(workspace_id) / "outputs"
        if outputs.is_symlink() or (outputs.exists() and not outputs.is_dir()):
            outputs.unlink(missing_ok=True)
        return self._make_dir(outputs)

    @staticmethod
    def resolve_path(workspace: Path, inputs: Path, path: str, *, write: bool) -> Path:
        posix = assert_safe_path(path)
        if path == "/workspace" or path.startswith("/workspace/"):
            relative = posix.relative_to("/workspace")
            root = workspace
        elif not write and (path == "/inputs" or path.startswith("/inputs/")):
            relative = posix.relative_to("/inputs")
            root = inputs
        else:
            raise PermissionError("permission_denied")
        resolved = (root / Path(*relative.parts)).resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise ValueError("invalid_path")
        return resolved

    @staticmethod
    def _directory_size(root: Path) -> int:
        return sum(
            path.stat().st_size
            for path in root.rglob("*")
            if not path.is_symlink() and path.is_file()
        )

    def enforce_limit(self, workspace_id: str) -> None:
        if self._directory_size(self.workspace_dir(workspace_id)) > self.limit_bytes:
            raise RuntimeError("Sandbox workspace exceeds its 5 GB limit")

    def stage_inputs(
        self,
        workspace_id: str,
        tenant_id: str,
        files: list[tuple[str, bytes]],
    ) -> None:
        if sum(len(content) for _path, content in files) > MAX_TRANSFER_BYTES:
            raise ValueError("Staged inputs exceed the 100 MB limit")
        self.ensure(workspace_id, tenant_id)
        inputs = self.inputs_dir(workspace_id)
        replacement = self._make_dir(inputs.with_name(f"inputs-{uuid4().hex}.tmp"))
        try:
            for path, content in files:
                if not path.startswith("/inputs/"):
                    raise ValueError("Only /inputs files may be staged")
                target = self.resolve_path(
                    self.workspace_dir(workspace_id), replacement, path, write=False
                )
                self._make_dir(target.parent)
                self._write(target, content)
            old = inputs.with_name(f"inputs-{uuid4().hex}.old")
            os.replace(inputs, old)
            os.replace(replacement, inputs)
            shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(replacement, ignore_errors=True)

    def upload(
        self,
        workspace_id: str,
        tenant_id: str,
        files: list[tuple[str, bytes]],
    ) -> list[WorkspaceFileResult]:
        self.ensure(workspace_id, tenant_id)
        results: list[WorkspaceFileResult] = []
        transferred = 0
        for path, content in files:
            try:
                transferred += len(content)
                if transferred > MAX_TRANSFER_BYTES:
                    raise ValueError("Uploaded files exceed the 100 MB transfer limit")
                target = self.resolve_path(
                    self.workspace_dir(workspace_id),
                    self.inputs_dir(workspace_id),
                    path,
                    write=True,
                )
                if target.exists():
                    results.append(WorkspaceFileResult(path, error="file already exists"))
                    continue
                self._make_dir(target.parent)
                self._write(target, content)
                results.append(WorkspaceFileResult(path))
            except (OSError, PermissionError, ValueError) as exc:
                results.append(WorkspaceFileResult(path, error=str(exc)))
        self.enforce_limit(workspace_id)
        return results

    def download(
        self,
        workspace_id: str,
        tenant_id: str,
        paths: list[str],
    ) -> list[WorkspaceFileResult]:
        self.ensure(workspace_id, tenant_id)
        results: list[WorkspaceFileResult] = []
        transferred = 0
        for path in paths:
            try:
                target = self.resolve_path(
                    self.workspace_dir(workspace_id),
                    self.inputs_dir(workspace_id),
                    path,
                    write=False,
                )
                if not target.is_file() or target.is_symlink():
                    results.append(WorkspaceFileResult(path, error="file_not_found"))
                    continue
                size = target.stat().st_size
                if transferred + size > MAX_TRANSFER_BYTES:
                    results.append(
                        WorkspaceFileResult(path, error="download transfer limit exceeded")
                    )
                    continue
                transferred += size
                results.append(WorkspaceFileResult(path, content=target.read_bytes()))
            except (OSError, PermissionError, ValueError) as exc:
                results.append(WorkspaceFileResult(path, error=str(exc)))
        return results

    def artifacts_since(
        self,
        workspace_id: str,
        started_at: datetime,
    ) -> WorkspaceArtifactScan:
        """Return accepted outputs and bounded rejection details for one execution."""
        output_dir = self.workspace_dir(workspace_id) / "outputs"
        if output_dir.is_symlink() or not output_dir.is_dir():
            return WorkspaceArtifactScan()
        output_root = output_dir.resolve()
        threshold = int(started_at.timestamp() * 1_000_000_000)
        names: list[str] = []
        dropped: list[tuple[str, str]] = []
        omitted_drops = 0
        images = 0

        def reject(name: str, reason: str) -> None:
            nonlocal omitted_drops
            if len(dropped) < MAX_FILE_ARTIFACTS_PER_EXECUTION:
                dropped.append((name, reason))
            else:
                omitted_drops += 1

        for path in sorted(output_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                continue
            if path.resolve().parent != output_root:
                continue
            stat = path.stat()
            if stat.st_mtime_ns < threshold:
                continue
            extension = path.suffix.lower()
            if extension not in COLLECTED_EXTENSIONS:
                reject(path.name, "Generated file type is not allowed")
                continue
            if extension == ".png":
                if stat.st_size > MAX_ARTIFACT_BYTES:
                    reject(path.name, "Generated PNG exceeds the 5 MB limit")
                    continue
                if images >= MAX_ARTIFACTS_PER_EXECUTION:
                    reject(
                        path.name,
                        f"This execution already has the maximum of "
                        f"{MAX_ARTIFACTS_PER_EXECUTION} plots",
                    )
                    continue
            elif stat.st_size > MAX_FILE_ARTIFACT_BYTES:
                reject(path.name, "Generated file exceeds the 10 MB limit")
                continue
            if len(names) >= MAX_FILE_ARTIFACTS_PER_EXECUTION:
                reject(
                    path.name,
                    f"This execution already has the maximum of "
                    f"{MAX_FILE_ARTIFACTS_PER_EXECUTION} generated files",
                )
                continue
            names.append(path.name)
            if extension == ".png":
                images += 1
        return WorkspaceArtifactScan(tuple(names), tuple(dropped), omitted_drops)

    def delete(self, workspace_id: str, tenant_id: str) -> None:
        root = self.workspace_root(workspace_id)
        if root.exists():
            self.ensure(workspace_id, tenant_id)
        shutil.rmtree(root, ignore_errors=True)

    def cleanup_expired(self, active_workspaces: set[str]) -> int:
        cutoff = time() - self.idle_ttl_seconds
        removed = 0
        if not self.root.is_dir():
            return 0
        for candidate in self.root.iterdir():
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or candidate.name in active_workspaces
                or candidate.stat().st_mtime >= cutoff
            ):
                continue
            shutil.rmtree(candidate, ignore_errors=True)
            removed += 1
        return removed
