"""Worker claim-loop, restart recovery, and failure containment."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from uuid import uuid4

from juena_core.sandbox.executor import ContainerCleanupError, ExecutionOutcome
from juena_core.sandbox.jobs import SandboxJobRecord
from juena_core.sandbox.worker import RESTART_MESSAGE, SandboxWorker


def _job(status: str = "running") -> SandboxJobRecord:
    now = datetime.now(UTC)
    return SandboxJobRecord(
        id=uuid4(),
        tenant_id="1" * 64,
        workspace_id="a" * 64,
        command="python -V",
        timeout_seconds=30,
        max_output_bytes=1000,
        status=status,  # type: ignore[arg-type]
        output="",
        exit_code=None,
        truncated=False,
        artifact_names=(),
        created_at=now,
        started_at=now,
        finished_at=None,
    )


class _Jobs:
    def __init__(self, queue: list[SandboxJobRecord]) -> None:
        self.queue = queue
        self.finished: list[dict] = []
        self.events: list[str] = []
        self.failed_active = 0
        self.claim_error: Exception | None = None
        self.finish_error: Exception | None = None

    def open(self) -> None:
        self.events.append("open")

    def close(self) -> None:
        self.events.append("close")

    def claim(self) -> SandboxJobRecord | None:
        if self.claim_error:
            raise self.claim_error
        return self.queue.pop(0) if self.queue else None

    def finish(self, job_id, **fields):
        if self.finish_error:
            raise self.finish_error
        self.finished.append({"id": job_id, **fields})
        self.events.append(f"finish:{fields['status']}")

    def fail_active(self, output: str) -> int:
        self.events.append("fail_active")
        self.failed_active = 1
        assert output == RESTART_MESSAGE
        return 1

    def fail_stale_queued(self) -> int:
        return 0

    def active_workspaces(self) -> set[str]:
        return set()

    def cleanup(self, *, older_than):
        return 0


class _Executor:
    def __init__(self, outcome: ExecutionOutcome | Exception) -> None:
        self.outcome = outcome
        self.events: list[str] = []
        self.settings = type("S", (), {"concurrency": 1})()

    def connect(self) -> None:
        self.events.append("connect")

    def validate_startup(self) -> None:
        self.events.append("validate")

    def remove_all(self) -> int:
        self.events.append("remove_all")
        return 2

    def close(self) -> None:
        self.events.append("close")

    def run(self, job):
        self.events.append("run")
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _worker(jobs: _Jobs, executor: _Executor, concurrency: int = 1) -> SandboxWorker:
    worker = SandboxWorker.__new__(SandboxWorker)
    worker.settings = type("S", (), {"concurrency": concurrency})()
    worker.jobs = jobs  # type: ignore[assignment]
    worker.executor = executor  # type: ignore[assignment]
    worker.workspaces = None  # type: ignore[assignment]
    return worker


def test_completed_job_is_committed_with_its_artifacts() -> None:
    job = _job()
    jobs = _Jobs([job])
    outcome = ExecutionOutcome("completed", "done\n", 0, False, ["plot.png"])
    worker = _worker(jobs, _Executor(outcome))

    worker._execute(job)

    assert jobs.finished == [
        {
            "id": job.id,
            "status": "completed",
            "output": "done\n",
            "exit_code": 0,
            "truncated": False,
            "artifact_names": ["plot.png"],
        }
    ]


def test_executor_failure_still_writes_a_terminal_row() -> None:
    job = _job()
    jobs = _Jobs([job])
    worker = _worker(jobs, _Executor(RuntimeError("podman is gone")))

    worker._execute(job)

    assert jobs.finished[0]["status"] == "failed"
    assert "podman is gone" in jobs.finished[0]["output"]


def test_slot_claims_until_stopped_and_never_reruns() -> None:
    job = _job()
    jobs = _Jobs([job])
    executor = _Executor(ExecutionOutcome("completed", "", 0, False, []))
    worker = _worker(jobs, executor)
    stop = threading.Event()

    def slot() -> None:
        worker._slot(stop, [])

    thread = threading.Thread(target=slot)
    thread.start()
    for _ in range(200):
        if jobs.finished:
            break
        threading.Event().wait(0.01)
    stop.set()
    thread.join(timeout=5)

    assert len(jobs.finished) == 1
    assert executor.events.count("run") == 1


def test_slot_failure_stops_the_worker() -> None:
    jobs = _Jobs([])
    jobs.claim_error = RuntimeError("database unavailable")
    worker = _worker(jobs, _Executor(ExecutionOutcome("completed", "", 0, False, [])))
    stop = threading.Event()
    fatal: list[BaseException] = []

    worker._slot(stop, fatal)

    assert stop.is_set()
    assert len(fatal) == 1
    assert "database unavailable" in str(fatal[0])


def test_terminal_update_failure_stops_the_worker_slot() -> None:
    jobs = _Jobs([_job()])
    jobs.finish_error = RuntimeError("database unavailable")
    worker = _worker(jobs, _Executor(ExecutionOutcome("completed", "", 0, False, [])))
    stop = threading.Event()
    fatal: list[BaseException] = []

    worker._slot(stop, fatal)

    assert stop.is_set()
    assert len(fatal) == 1


def test_cleanup_failure_is_terminal_after_job_is_failed() -> None:
    job = _job()
    jobs = _Jobs([job])
    worker = _worker(jobs, _Executor(ContainerCleanupError("remove failed")))

    try:
        worker._execute(job)
    except ContainerCleanupError:
        pass
    else:
        raise AssertionError("cleanup failure did not escape the job boundary")

    assert jobs.finished[0]["status"] == "failed"


def test_startup_removes_containers_and_fails_interrupted_jobs() -> None:
    jobs = _Jobs([])
    executor = _Executor(ExecutionOutcome("completed", "", 0, False, []))
    worker = _worker(jobs, executor)
    stop = threading.Event()
    stop.set()

    worker.run(stop)

    # Recovery happens before any slot can claim: nothing is reattached, and the
    # previous life's running jobs are failed rather than rerun.
    assert executor.events[:3] == ["connect", "validate", "remove_all"]
    assert jobs.failed_active == 1
    assert jobs.finished == []
