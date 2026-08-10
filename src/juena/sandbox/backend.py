"""Deep Agents backends that delegate execution to the sandbox worker."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    _build_glob_cmd,
    _build_grep_cmd,
    _parse_glob_output,
    _parse_grep_output,
)
from langgraph.config import get_stream_writer

from juena.core.config import global_config
from juena.sandbox.artifacts import get_artifact_store
from juena.sandbox.constants import MAX_FILE_ARTIFACTS_PER_EXECUTION
from juena.sandbox.jobs import SandboxJobs
from juena.sandbox.policy import is_redundant_artifact_export
from juena.sandbox.workspace import (
    WorkspaceFileResult,
    WorkspaceStore,
    assert_safe_path,
)

# Every status the backend can report, including the two the sandbox returns
# without ever writing a row.
STATUS_LABELS = {
    "queued": "Sandbox command queued",
    "running": "Sandbox command running",
    "completed": "Sandbox command completed",
    "timed_out": "Sandbox command timed out",
    "failed": "Sandbox command failed",
    "busy": "Sandbox capacity is busy",
    "unavailable": "Sandbox worker unavailable",
}


class PodmanSandboxBackend(BaseSandbox):
    """Tenant-bound backend whose commands run only through the host worker."""

    def __init__(
        self,
        *,
        jobs: SandboxJobs,
        workspaces: WorkspaceStore,
        tenant_id: str,
        workspace_id: str,
        user_id: str,
        thread_id: str,
        run_id: str | None,
    ) -> None:
        self._jobs = jobs
        self._workspaces = workspaces
        self._tenant_id = tenant_id
        self._workspace_id = workspace_id
        self._user_id = user_id
        self._thread_id = thread_id
        self._run_id = run_id

    @property
    def id(self) -> str:
        return self._workspace_id

    @staticmethod
    def _status(status: str, label: str, **extra: Any) -> None:
        try:
            get_stream_writer()(
                {"type": "sandbox_status", "status": status, "label": label, **extra}
            )
        except (KeyError, RuntimeError):
            # Backend unit tests and non-streaming invocations have no writer.
            return

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        effective_timeout = min(
            timeout or global_config.SANDBOX_EXECUTION_TIMEOUT_SECONDS,
            global_config.SANDBOX_EXECUTION_TIMEOUT_SECONDS,
        )
        store = get_artifact_store()
        if is_redundant_artifact_export(command):
            store.audit(
                {
                    "event": "sandbox_execution_skipped",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "command": command,
                    "reason": "artifact_already_collected",
                }
            )
            return ExecuteResponse(
                output=(
                    "The application already collects files under /workspace/outputs "
                    "and displays PNG artifacts automatically. Do not encode or print "
                    "the file; finish the report using the collected artifact metadata."
                ),
                exit_code=0,
            )
        try:
            result = self._jobs.execute(
                tenant_id=self._tenant_id,
                workspace_id=self._workspace_id,
                command=command,
                timeout_seconds=effective_timeout,
                max_output_bytes=global_config.SANDBOX_MAX_OUTPUT_BYTES,
                status_callback=lambda status: self._status(
                    status,
                    STATUS_LABELS.get(status, "Sandbox command running"),
                ),
            )
            refs = []
            paths = [f"/workspace/outputs/{name}" for name in result.artifact_names]
            for artifact in self._collect(paths):
                if artifact.error or artifact.content is None:
                    continue
                try:
                    refs.append(
                        store.register_artifact(
                            user_id=self._user_id,
                            thread_id=self._thread_id,
                            run_id=self._run_id,
                            filename=PurePosixPath(artifact.path).name,
                            content=artifact.content,
                        )
                    )
                except (OSError, ValueError):
                    continue
            store.audit(
                {
                    "event": "sandbox_execution",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "job_id": str(result.job_id),
                    "command": command,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "artifact_ids": [ref.artifact_id for ref in refs],
                }
            )
            self._status(
                result.status,
                STATUS_LABELS.get(result.status, "Sandbox command failed"),
                job_id=str(result.job_id),
                exit_code=result.exit_code,
            )
            model_output = result.output.rstrip()
            if refs:
                artifact_lines = [
                    (
                        f"- {ref.artifact_id} | {ref.filename} | "
                        f"{ref.mime_type} | {ref.caption}"
                    )
                    for ref in refs
                ]
                artifact_note = (
                    "Artifacts collected automatically by the application:\n"
                    + "\n".join(artifact_lines)
                    + "\nDo not run another command to inspect, encode, print, or "
                    "verify these files. Finish the report now."
                )
                model_output = "\n\n".join(
                    part for part in (model_output, artifact_note) if part
                )
            return ExecuteResponse(
                output=model_output,
                exit_code=result.exit_code,
                truncated=result.truncated,
            )
        except Exception as exc:  # noqa: BLE001 - keep chat alive on sandbox outages
            store.audit(
                {
                    "event": "sandbox_execution",
                    "user_id": self._user_id,
                    "thread_id": self._thread_id,
                    "run_id": self._run_id,
                    "command": command,
                    "status": "worker_error",
                }
            )
            self._status("failed", "Sandbox worker unavailable")
            return ExecuteResponse(output=f"Sandbox unavailable: {exc}", exit_code=None)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        search_path = path or "/workspace"
        if PurePosixPath(search_path) == PurePosixPath("/"):
            return GlobResult(error="Recursive searches from / are not allowed")
        result = self.execute(_build_glob_cmd(pattern, search_path), timeout=15)
        if result.exit_code not in (0, None):
            return GlobResult(error=result.output)
        return _parse_glob_output(result.output, search_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        search_path = path or "/workspace"
        if PurePosixPath(search_path) == PurePosixPath("/"):
            return GrepResult(error="Recursive searches from / are not allowed")
        result = self.execute(_build_grep_cmd(pattern, search_path, glob), timeout=15)
        if result.exit_code not in (0, 1, None):
            return GrepResult(error=result.output)
        return _parse_grep_output(result, search_path)

    def _collect(self, paths: list[str]) -> list[WorkspaceFileResult]:
        """Read finished outputs under the same guard as explicit file access."""

        if not paths:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return []
            return self._workspaces.download(self._workspace_id, self._tenant_id, paths)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if not files:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return [
                    FileUploadResponse(path=path, error="workspace busy")
                    for path, _content in files
                ]
            workspace_results = self._workspaces.upload(
                self._workspace_id,
                self._tenant_id,
                files,
            )
        responses = [
            FileUploadResponse(
                path=path,
                error=(
                    workspace_results[index].error
                    if index < len(workspace_results)
                    else "missing workspace response"
                ),
            )
            for index, (path, _content) in enumerate(files)
        ]
        store = get_artifact_store()
        registered = 0
        for (path, content), response in zip(files, responses, strict=False):
            if response.error or not path.startswith("/workspace/outputs/"):
                continue
            try:
                store.register_artifact(
                    user_id=self._user_id,
                    thread_id=self._thread_id,
                    run_id=self._run_id,
                    filename=PurePosixPath(path).name,
                    content=content,
                )
            except (ValueError, OSError):
                continue
            registered += 1
            if registered >= MAX_FILE_ARTIFACTS_PER_EXECUTION:
                break
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if not paths:
            return []
        with self._jobs.workspace_guard(self._workspace_id) as available:
            if not available:
                return [FileDownloadResponse(path=path, error="workspace busy") for path in paths]
            workspace_results = self._workspaces.download(
                self._workspace_id,
                self._tenant_id,
                paths,
            )
        responses: list[FileDownloadResponse] = []
        for index, path in enumerate(paths):
            if index >= len(workspace_results):
                responses.append(FileDownloadResponse(path=path, error="missing workspace response"))
                continue
            item = workspace_results[index]
            responses.append(
                FileDownloadResponse(path=path, content=item.content, error=item.error)
            )
        return responses


class ReadOnlySandboxInputsBackend(BackendProtocol):
    """Route-local `/inputs` adapter over the sandbox's read-only input mount."""

    def __init__(self, sandbox: SandboxBackendProtocol) -> None:
        self._sandbox = sandbox

    @staticmethod
    def _full(path: str | None) -> str:
        local = path or "/"
        assert_safe_path(local, "Invalid /inputs path")
        return "/inputs" if local == "/" else f"/inputs{local}"

    @staticmethod
    def _local(path: str) -> str:
        if path == "/inputs":
            return "/"
        if path.startswith("/inputs/"):
            return path[len("/inputs") :]
        return path

    def ls(self, path: str) -> LsResult:
        try:
            result = self._sandbox.ls(self._full(path))
        except ValueError as exc:
            return LsResult(error=str(exc))
        entries = [
            FileInfo(**{**item, "path": self._local(item["path"])})
            for item in (result.entries or [])
        ]
        return LsResult(error=result.error, entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            return self._sandbox.read(self._full(file_path), offset=offset, limit=limit)
        except ValueError as exc:
            return ReadResult(error=str(exc))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        try:
            result = self._sandbox.grep(pattern, self._full(path), glob)
        except ValueError as exc:
            return GrepResult(error=str(exc))
        matches = [
            GrepMatch(**{**item, "path": self._local(item["path"])})
            for item in (result.matches or [])
        ]
        return GrepResult(error=result.error, matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        try:
            result = self._sandbox.glob(pattern, self._full(path))
        except ValueError as exc:
            return GlobResult(error=str(exc))
        matches = [
            FileInfo(**{**item, "path": self._local(item["path"])})
            for item in (result.matches or [])
        ]
        return GlobResult(error=result.error, matches=matches)

    @staticmethod
    def write(file_path: str, content: str) -> WriteResult:
        del content
        return WriteResult(error=f"Cannot write to read-only staged input path '{file_path}'.")

    @staticmethod
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        del old_string, new_string, replace_all
        return EditResult(error=f"Cannot edit read-only staged input path '{file_path}'.")

    @staticmethod
    def upload_files(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [
            FileUploadResponse(path=path, error="permission_denied") for path, _content in files
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        full_paths: list[str] = []
        for path in paths:
            try:
                full_paths.append(self._full(path))
            except ValueError as exc:
                return [FileDownloadResponse(path=item, error=str(exc)) for item in paths]
        results = self._sandbox.download_files(full_paths)
        return [
            FileDownloadResponse(path=path, content=result.content, error=result.error)
            for path, result in zip(paths, results, strict=False)
        ]
