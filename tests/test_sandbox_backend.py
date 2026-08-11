"""Deep Agents backend routing and queued execution tests."""

import io
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from deepagents.backends.protocol import BackendProtocol, LsResult
from deepagents.middleware.filesystem import FilesystemMiddleware, supports_execution
from langchain_core.messages import HumanMessage
from PIL import Image

# Bound at import time, exactly as the sandbox modules under test bind it.
# tests/test_saml_auth.py reloads juena.core.config and installs a new
# global_config; resolving it freshly here would patch an object the code
# under test is not reading.
from juena.core.config import global_config
from juena.schema.agents import SpecialistReport
from juena.agents.specialist_outcome import SpecialistOutcomeMiddleware
from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.backend import PodmanSandboxBackend, ReadOnlySandboxInputsBackend
from juena.sandbox.constants import MAX_IMAGE_ARTIFACTS_PER_TURN
from juena.sandbox.evidence import capture_sandbox_execution
from juena.sandbox.jobs import SandboxExecutionResult, workspace_id_for
from juena.sandbox.runtime import (
    RuntimePodmanSandboxBackend,
    build_sandbox_backend,
    set_sandbox_jobs_for_tests,
    set_workspace_store_for_tests,
)
from juena.sandbox.workspace import WorkspaceStore


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class _Jobs:
    def __init__(self, workspaces: WorkspaceStore) -> None:
        self.requests = []
        self.workspaces = workspaces

    @contextmanager
    def workspace_guard(self, _workspace_id):
        yield True

    def execute(self, **request):
        self.requests.append(request)
        output_dir = self.workspaces.workspace_dir(request["workspace_id"]) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.py").write_text("print('done')\n", encoding="utf-8")
        return SandboxExecutionResult(
            job_id=uuid4(),
            output="finished\n",
            exit_code=0,
            status="completed",
            artifact_names=("analysis.py",),
        )


def test_execute_returns_command_output_and_collected_file(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    workspaces.ensure("a" * 64, "1" * 64)
    jobs = _Jobs(workspaces)
    backend = PodmanSandboxBackend(
        jobs=jobs,  # type: ignore[arg-type]
        workspaces=workspaces,
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )

    try:
        with capture_sandbox_execution() as evidence:
            result = backend.execute("python plot.py", timeout=900)
        refs = store.claim_for_message("user-a", "thread-a")
    finally:
        set_artifact_store_for_tests(None)

    assert result.exit_code == 0
    assert (
        jobs.requests[0]["timeout_seconds"]
        == global_config.SANDBOX_EXECUTION_TIMEOUT_SECONDS
    )
    assert "Artifacts collected automatically" in result.output
    assert "analysis.py" in result.output
    assert len(evidence) == 1
    assert evidence[0].succeeded is True
    assert evidence[0].artifact_filenames == ["analysis.py"]
    assert evidence[0].artifact_ids == [refs[0]["artifact_id"]]
    # The approval record is the middleware's job, not the backend's.
    assert [item["filename"] for item in refs] == ["analysis.py"]


class _PlotJobs(_Jobs):
    """Writes a PNG to outputs rather than the default script."""

    def execute(self, **request):
        self.requests.append(request)
        output_dir = self.workspaces.workspace_dir(request["workspace_id"]) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot.png").write_bytes(_png())
        return SandboxExecutionResult(
            job_id=uuid4(),
            output="fit converged\n",
            exit_code=0,
            status="completed",
            artifact_names=("plot.png",),
        )


def _plot_backend(tmp_path, jobs_class=_PlotJobs):
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    workspaces.ensure("a" * 64, "1" * 64)
    jobs = jobs_class(workspaces)
    return PodmanSandboxBackend(
        jobs=jobs,  # type: ignore[arg-type]
        workspaces=workspaces,
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )


def test_dropped_plot_is_reported_to_the_model(tmp_path) -> None:
    """The bug that made the agent claim a figure the user never saw.

    A refused registration used to be swallowed, leaving the model a clean exit
    code and its own script's success message to reason from.
    """
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    for _ in range(MAX_IMAGE_ARTIFACTS_PER_TURN):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="earlier.png",
            content=_png(),
        )
    backend = _plot_backend(tmp_path)

    try:
        result = backend.execute("python fit.py")
    finally:
        set_artifact_store_for_tests(None)

    assert "fit converged" in result.output
    assert "NOT delivered" in result.output
    assert "plot.png" in result.output
    assert f"maximum of {MAX_IMAGE_ARTIFACTS_PER_TURN} plots" in result.output
    assert "Artifacts collected automatically" not in result.output


def test_busy_workspace_reports_the_missing_file(tmp_path) -> None:
    """_collect returns nothing when the guard is busy, which used to be silent."""

    class _BusyJobs(_PlotJobs):
        @contextmanager
        def workspace_guard(self, _workspace_id):
            # Busy only for the post-run collection; execute() never asks.
            yield False

    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    backend = _plot_backend(tmp_path, _BusyJobs)

    try:
        result = backend.execute("python fit.py")
    finally:
        set_artifact_store_for_tests(None)

    assert "plot.png" in result.output
    assert "workspace was busy" in result.output


class _InputSandbox:
    def __init__(self) -> None:
        self.paths = []

    def ls(self, path):
        self.paths.append(path)
        return LsResult(entries=[{"path": "/inputs/data.csv", "is_dir": False}])


def test_inputs_adapter_restores_prefix_and_rejects_mutation() -> None:
    sandbox = _InputSandbox()
    adapter = ReadOnlySandboxInputsBackend(sandbox)  # type: ignore[arg-type]

    result = adapter.ls("/")

    assert sandbox.paths == ["/inputs"]
    assert result.entries == [{"path": "/data.csv", "is_dir": False}]
    assert adapter.write("/data.csv", "changed").error
    assert adapter.edit("/data.csv", "a", "b").error
    assert adapter.upload_files([("/data.csv", b"changed")])[0].error == "permission_denied"


def test_runtime_backend_routes_only_explicit_read_only_prefixes(monkeypatch, tmp_path) -> None:
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    jobs = _Jobs(workspaces)
    set_sandbox_jobs_for_tests(jobs)  # type: ignore[arg-type]
    set_workspace_store_for_tests(workspaces)
    monkeypatch.setattr(global_config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(global_config, "SANDBOX_IDENTITY_SECRET", "t" * 32)
    monkeypatch.setattr(global_config, "SANDBOX_WORKSPACE_ROOT", workspaces.root)
    runtime = SimpleNamespace(
        context=SimpleNamespace(user_id="user-a", thread_id="thread-a"),
        config={"run_id": "run-1"},
    )
    monkeypatch.setattr("juena.sandbox.runtime.get_runtime", lambda: runtime)
    composite = build_sandbox_backend(
        repo_cache_root=None,
        skills_dir=None,
        has_skills=False,
    )

    assert isinstance(composite, BackendProtocol)
    assert not callable(composite)
    assert composite.artifacts_root == "/workspace"

    try:
        middleware = FilesystemMiddleware(backend=composite)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            resolved = middleware._get_backend(runtime)
        assert resolved is composite
        assert supports_execution(composite)
        assert isinstance(composite.default, RuntimePodmanSandboxBackend)
        assert composite.artifacts_root == "/workspace"
        assert middleware._large_tool_results_prefix == "/workspace/large_tool_results"
        first_workspace = workspace_id_for("user-a", "thread-a", "t" * 32)
        assert composite.default.id == first_workspace

        runtime.context.user_id = "user-b"
        assert composite.default.id == workspace_id_for("user-b", "thread-a", "t" * 32)
        assert composite.default.id != first_workspace

        inputs_backend, stripped = composite._get_backend_and_key("/inputs/data.csv")
        assert isinstance(inputs_backend, ReadOnlySandboxInputsBackend)
        assert stripped == "/data.csv"

        for path in (
            "/",
            "/tmp/file",
            "/etc/passwd",
            "relative.py",
            "../escape",
            "/large_tool_results/result.txt",
            "/conversation_history/history.txt",
        ):
            backend, _key = composite._get_backend_and_key(path)
            assert backend is composite.default
    finally:
        set_sandbox_jobs_for_tests(None)
        set_workspace_store_for_tests(None)


def test_redundant_base64_export_is_skipped_without_using_worker(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    jobs = _Jobs(workspaces)
    backend = PodmanSandboxBackend(
        jobs=jobs,  # type: ignore[arg-type]
        workspaces=workspaces,
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )

    try:
        result = backend.execute(
            "base64 /workspace/outputs/plot.png",
            timeout=30,
        )
    finally:
        set_artifact_store_for_tests(None)

    assert result.exit_code == 0
    assert "already collects" in result.output
    assert jobs.requests == []


def test_workspace_identity_changes_for_user_or_thread() -> None:
    token = "t" * 32
    first = workspace_id_for("user-a", "thread-a", token)

    assert first == workspace_id_for("user-a", "thread-a", token)
    assert first != workspace_id_for("user-b", "thread-a", token)
    assert first != workspace_id_for("user-a", "thread-b", token)


def test_root_search_is_rejected_and_later_write_remains_available(tmp_path) -> None:
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    jobs = _Jobs(workspaces)
    backend = PodmanSandboxBackend(
        jobs=jobs,  # type: ignore[arg-type]
        workspaces=workspaces,
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )

    assert backend.glob("**/*", "/./").error == "Recursive searches from / are not allowed"
    assert jobs.requests == []
    assert backend.upload_files([("/workspace/later.txt", b"responsive")])[0].error is None


def test_output_upload_reports_when_the_file_was_not_delivered(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    workspaces = WorkspaceStore(tmp_path / "workspaces")
    jobs = _Jobs(workspaces)
    backend = PodmanSandboxBackend(
        jobs=jobs,  # type: ignore[arg-type]
        workspaces=workspaces,
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )

    try:
        write = backend.write("/workspace/outputs/empty.txt", "")
        package = SpecialistOutcomeMiddleware(
            specialist_name="software-specialist"
        ).after_agent(
            {
                "messages": [HumanMessage("Create empty.txt.")],
                "structured_response": SpecialistReport(
                    status="completed",
                    finding="The requested report is ready.",
                ),
            },
            SimpleNamespace(
                context=SimpleNamespace(user_id="user-a", thread_id="thread-a"),
                config={"run_id": "run-1"},
            ),
        )["messages"][0].text
    finally:
        set_artifact_store_for_tests(None)

    # The write itself succeeded, so reporting an error here would be false and
    # would send the model into a rewrite that collides with its own file.
    assert write.error is None
    assert write.path == "/workspace/outputs/empty.txt"
    assert "Produced but NOT delivered" in package
    assert "empty.txt" in package
    delivered = package.split("Result artifacts delivered:")[1].splitlines()[0]
    assert "empty.txt" not in delivered
    attached = store.claim_for_message("user-a", "thread-a")
    assert "empty.txt" not in {item["filename"] for item in attached}
    assert '"event": "artifact_dropped"' in (tmp_path / "audit.jsonl").read_text()
