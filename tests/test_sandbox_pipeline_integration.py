"""Opt-in end-to-end tests against a running Postgres and Podman worker."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena_core.sandbox.backend import PodmanSandboxBackend
from juena_core.sandbox.constants import ACTIVE_STATUSES
from juena_core.sandbox.jobs import SandboxJobs
from juena_core.sandbox.workspace import WorkspaceStore

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


def test_jscatter_fit_graceplot_and_artifact_reload(tmp_path) -> None:
    jobs = _jobs()
    workspaces = WorkspaceStore(Path(WORKSPACE_ROOT))  # type: ignore[arg-type]
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    user_id = "jscatter-live-user"
    thread_id = f"jscatter-live-{uuid4().hex}"
    backend = PodmanSandboxBackend(
        jobs=jobs,
        workspaces=workspaces,
        tenant_id=_opaque_id(),
        workspace_id=_opaque_id(),
        user_id=user_id,
        thread_id=thread_id,
        run_id="jscatter-live-run",
    )
    command = r'''mkdir -p outputs && python - <<'PY'
from pathlib import Path
import json
import numpy as np
import jscatter as js

js.headless(True)
print("JScatter", js.__version__)
assert js.__version__ == "1.9.0.1"

def sphere_model(q, radius, scale, background):
    form = js.ff.sphere(q=q, radius=radius, contrast=0.01)
    return scale * form.Y + background

q = np.geomspace(0.05, 1.2, 80)
truth = sphere_model(q, radius=5.5, scale=1.8, background=0.04)
uncertainty = np.maximum(0.02 * truth, 0.02)
rng = np.random.default_rng(20260811)
measured = truth + rng.normal(scale=uncertainty)
data = js.dA(np.vstack([q, measured, uncertainty]), XYeYeX=[0, 1, 2])
assert np.all(np.isfinite(data.eY)) and np.all(data.eY > 0)

data.fit(
    sphere_model,
    freepar={"radius": 5.0, "scale": 1.5, "background": 0.03},
    fixpar={},
    mapNames={"q": "X"},
    method="lm",
    output=False,
)
fit = data.lastfit
assert np.isfinite(fit.radius) and np.isfinite(fit.radius_err)
assert abs(fit.radius - 5.5) < 0.5
fit.savetxt("outputs/jscatter-fit.dat")

plot = js.GracePlot(headless=True)
plot.plot(data, sy=1, li=0, le="synthetic data")
plot.plot(fit, sy=0, li=1, le="JScatter fit")
plot.xaxis(label=r"q / nm\S-1\N")
plot.yaxis(label="I(q)")
plot.legend()
plot.save("outputs/jscatter-fit.png", size=(3.4, 2.4), dpi=150)
plot.exit()

Path("outputs/jscatter-fit.json").write_text(
    json.dumps(
        {
            "jscatter_version": js.__version__,
            "radius": float(fit.radius),
            "radius_error": float(fit.radius_err),
        },
        indent=2,
    )
)
print("fit radius", fit.radius, "+/-", fit.radius_err)
PY'''

    try:
        result = backend.execute(command, timeout=60)
        pending = store.peek_result_refs(user_id, thread_id)
        delivered = store.claim_for_message(user_id, thread_id)
        reloaded = [store.get(user_id, item["artifact_id"]) for item in delivered]
    finally:
        set_artifact_store_for_tests(None)
        jobs.close()

    assert result.exit_code == 0, result.output
    assert "JScatter 1.9.0.1" in result.output
    assert "fit radius" in result.output
    assert {item.filename for item in pending} == {
        "jscatter-fit.dat",
        "jscatter-fit.json",
        "jscatter-fit.png",
    }
    assert {item["filename"] for item in delivered} == {
        "jscatter-fit.dat",
        "jscatter-fit.json",
        "jscatter-fit.png",
    }
    assert all(item is not None for item in reloaded)
    png = next(
        item for item in reloaded if item is not None and item[0].filename.endswith(".png")
    )
    assert png[1].startswith(b"\x89PNG\r\n\x1a\n")


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
