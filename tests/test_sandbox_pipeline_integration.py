"""Opt-in end-to-end tests against a running Postgres and Podman worker."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.backend import PodmanSandboxBackend
from juena.sandbox.constants import ACTIVE_STATUSES
from juena.sandbox.jobs import SandboxJobs
from juena.sandbox.workspace import WorkspaceStore

DATABASE_URL = os.getenv("JUENA_TEST_DATABASE_URL")
WORKSPACE_ROOT = os.getenv("JUENA_TEST_WORKSPACE_ROOT")
CONCURRENCY = int(os.getenv("JUENA_TEST_SANDBOX_CONCURRENCY", "4"))
pytestmark = pytest.mark.skipif(
    not all((DATABASE_URL, WORKSPACE_ROOT)),
    reason="set JUENA_TEST_DATABASE_URL and JUENA_TEST_WORKSPACE_ROOT to run the "
    "sandbox pipeline tests against a live worker",
)


def _opaque_id() -> str:
    return uuid4().hex + uuid4().hex


def _jobs() -> SandboxJobs:
    jobs = SandboxJobs(DATABASE_URL, concurrency=CONCURRENCY)  # type: ignore[arg-type]
    jobs.open()
    return jobs


def _active_count(jobs: SandboxJobs, workspace_ids: list[str]) -> int:
    with jobs.pool.connection() as connection:
        row = connection.execute(
            "SELECT count(*) AS count FROM sandbox_jobs "
            "WHERE workspace_id = ANY(%s) AND status = ANY(%s)",
            (workspace_ids, list(ACTIVE_STATUSES)),
        ).fetchone()
    return row["count"] if row else 0


def _wait_for_active(jobs: SandboxJobs, workspace_ids: list[str], expected: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _active_count(jobs, workspace_ids) == expected:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"Expected {expected} active jobs, found {_active_count(jobs, workspace_ids)}"
    )


def test_worker_output_and_artifacts_reach_the_application_validator(tmp_path) -> None:
    jobs = _jobs()
    workspaces = WorkspaceStore(Path(WORKSPACE_ROOT))  # type: ignore[arg-type]
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    backend = PodmanSandboxBackend(
        jobs=jobs,
        workspaces=workspaces,
        tenant_id=_opaque_id(),
        workspace_id=_opaque_id(),
        user_id="live-user",
        thread_id="live-thread",
        run_id="live-run",
    )
    command = (
        "mkdir -p outputs && python -c \"from pathlib import Path; "
        "from PIL import Image; import sys; print('stdout ok'); "
        "print('stderr ok', file=sys.stderr); "
        "Path('outputs/data.csv').write_text('x,y\\n1,2\\n'); "
        "Image.new('RGB',(8,8),'red').save('outputs/plot.png')\""
    )

    try:
        result = backend.execute(command, timeout=30)
        artifacts = store.claim_for_message("live-user", "live-thread")
    finally:
        set_artifact_store_for_tests(None)
        jobs.close()

    assert result.exit_code == 0
    assert "stdout ok" in result.output
    assert "stderr ok" in result.output
    assert {item["filename"] for item in artifacts} == {"data.csv", "plot.png"}
    assert {item["kind"] for item in artifacts} == {"file", "image"}


def test_slots_fill_then_the_next_submission_is_busy_without_a_row() -> None:
    jobs = _jobs()
    tenant_ids = [_opaque_id() for _index in range(CONCURRENCY)]
    workspace_ids = [_opaque_id() for _index in range(CONCURRENCY)]

    def execute(index: int):
        return jobs.execute(
            tenant_id=tenant_ids[index],
            workspace_id=workspace_ids[index],
            command="python -c \"import time; time.sleep(4); print('done')\"",
            timeout_seconds=20,
            max_output_bytes=1000,
        )

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(execute, index) for index in range(CONCURRENCY)]
            _wait_for_active(jobs, workspace_ids, CONCURRENCY)

            overflow = jobs.execute(
                tenant_id=_opaque_id(),
                workspace_id=_opaque_id(),
                command="echo overflow",
                timeout_seconds=10,
                max_output_bytes=1000,
            )
            assert overflow.status == "busy"
            # `busy` is a return value, never a row holding a slot.
            assert overflow.job_id is None

            duplicate = jobs.execute(
                tenant_id=tenant_ids[0],
                workspace_id=_opaque_id(),
                command="echo duplicate",
                timeout_seconds=10,
                max_output_bytes=1000,
            )
            assert duplicate.status == "busy"
            assert all(future.result().status == "completed" for future in futures)
    finally:
        jobs.close()


def test_timed_out_job_does_not_block_another_tenant() -> None:
    jobs = _jobs()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            endless = pool.submit(
                jobs.execute,
                tenant_id=_opaque_id(),
                workspace_id=_opaque_id(),
                command="python -c \"while True: pass\"",
                timeout_seconds=2,
                max_output_bytes=1000,
            )
            quick = pool.submit(
                jobs.execute,
                tenant_id=_opaque_id(),
                workspace_id=_opaque_id(),
                command="echo independent",
                timeout_seconds=10,
                max_output_bytes=1000,
            )
            assert quick.result().status == "completed"
            assert endless.result().status == "timed_out"
    finally:
        jobs.close()


def test_submission_without_a_worker_reports_unavailable() -> None:
    """Run this with the worker stopped."""

    if os.getenv("JUENA_TEST_WORKER_STOPPED") != "1":
        pytest.skip("set JUENA_TEST_WORKER_STOPPED=1 with the worker stopped")
    jobs = _jobs()
    try:
        result = jobs.execute(
            tenant_id=_opaque_id(),
            workspace_id=_opaque_id(),
            command="echo unreachable",
            timeout_seconds=10,
            max_output_bytes=1000,
        )
        assert result.status == "unavailable"
        assert result.job_id is None
    finally:
        jobs.close()
