"""Deep Agents backend routing and broker delegation tests."""

import base64
import warnings
from types import SimpleNamespace

import pytest
from deepagents.backends.protocol import BackendProtocol, LsResult
from deepagents.middleware.filesystem import FilesystemMiddleware, supports_execution

from juena.core.config import global_config
from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.backend import PodmanSandboxBackend, ReadOnlySandboxInputsBackend
from juena.sandbox.broker_client import set_broker_client_for_tests, workspace_id_for
from juena.sandbox.runtime import RuntimePodmanSandboxBackend, build_sandbox_backend
from juena.sandbox.schemas import BrokerArtifact, ExecuteResult


class _Broker:
    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return ExecuteResult(
            job_id="job-1",
            output="finished\n",
            exit_code=0,
            status="completed",
            artifacts=[
                BrokerArtifact(
                    filename="analysis.py",
                    content_base64=base64.b64encode(b"print('done')\n").decode("ascii"),
                )
            ],
        )


def test_execute_returns_command_output_and_collected_file(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    broker = _Broker()
    backend = PodmanSandboxBackend(
        broker=broker,
        workspace_id="a" * 64,
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
    )

    try:
        result = backend.execute("python plot.py", timeout=900)
        refs = store.claim_for_message("user-a", "thread-a")
    finally:
        set_artifact_store_for_tests(None)

    assert result.exit_code == 0
    assert broker.requests[0].timeout_seconds == global_config.SANDBOX_EXECUTION_TIMEOUT_SECONDS
    assert "Artifacts collected automatically" in result.output
    assert "analysis.py" in result.output
    # The approval record is the middleware's job, not the backend's.
    assert [item["filename"] for item in refs] == ["analysis.py"]


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


def test_runtime_backend_routes_only_explicit_read_only_prefixes(monkeypatch) -> None:
    broker = _Broker()
    set_broker_client_for_tests(broker)  # type: ignore[arg-type]
    monkeypatch.setattr(global_config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(global_config, "SANDBOX_BROKER_URL", "http://broker")
    monkeypatch.setattr(global_config, "SANDBOX_BROKER_TOKEN", "t" * 32)
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
        set_broker_client_for_tests(None)


def test_redundant_base64_export_is_skipped_without_using_broker(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    broker = _Broker()
    backend = PodmanSandboxBackend(
        broker=broker,
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
    assert broker.requests == []


def test_workspace_identity_changes_for_user_or_thread() -> None:
    token = "t" * 32
    first = workspace_id_for("user-a", "thread-a", token)

    assert first == workspace_id_for("user-a", "thread-a", token)
    assert first != workspace_id_for("user-b", "thread-a", token)
    assert first != workspace_id_for("user-a", "thread-b", token)
