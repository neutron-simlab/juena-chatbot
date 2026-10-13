"""Run and recover unattended research from Postgres.

The job row is the queue. Each job holds a database session lock and writes its
LangGraph checkpoints through that same connection. A process restart releases
the lock, so another runner can continue from the last saved step. Shutdown
leaves unfinished jobs active; it never fails another process's work.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from juena_core.agents.findings import FindingsMap, findings_delta, fingerprint
from juena_core.agents.specialist_outcome import outcome_verified
from juena_core.log import get_logger
from juena.research import jobs
from juena.research.constants import DEFAULT_TIMEOUT_SECONDS, TIMEOUT_MESSAGE
from juena.server.database.models import ResearchJob
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena_core.tracing import with_tracing

logger = get_logger(__name__)

NO_REPORT = (
    "the specialist finished without producing a verified report, so nothing it "
    "did mid-run can be treated as a result"
)
POLL_SECONDS = 2
_running: dict[UUID, asyncio.Task] = {}


def _final_package(result: dict[str, Any]) -> AIMessage | None:
    """The specialist's final hand-off, including the server's verdict."""
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage) and message.text and message.text.rstrip():
            return message
    return None


def _is_result(package: AIMessage | None) -> bool:
    # Failed specialists also return nonempty messages. Only the middleware's
    # boolean verdict authorizes publishing the report and its findings.
    return package is not None and outcome_verified(package)


def _written_findings(result: dict[str, Any], seed: dict[str, Any]) -> FindingsMap:
    """Return changes against the original input, including after recovery."""
    return findings_delta(seed, result.get("files"))


async def _invoke(runnable: Any, job: ResearchJob) -> dict[str, Any]:
    config = with_tracing(
        {"configurable": {"thread_id": f"research:{job.id}"}},
        thread_id=job.thread_id,
        user_id=str(job.user_id),
    )
    snapshot = await runnable.aget_state(config)
    if snapshot.created_at and not snapshot.next:
        # The graph finished, but the process died before recording its result.
        return snapshot.values
    # Recovery never extends the original deadline. A finished checkpoint above
    # is still delivered even when the service was offline past that deadline.
    remaining = DEFAULT_TIMEOUT_SECONDS - (datetime.now(UTC) - job.started_at).total_seconds()
    if remaining <= 0:
        raise TimeoutError
    initial = None if snapshot.created_at else {
        "messages": [HumanMessage(content=job.objective)],
        "files": job.run_input["files"],
    }
    async with asyncio.timeout(remaining):
        return await runnable.ainvoke(
            initial,
            config=config,
            context=RuntimeModelContext(
                provider=job.run_input["provider"],
                model=job.run_input["model"],
                thread_id=job.thread_id,
                user_id=str(job.user_id),
                run_id=f"research:{job.id}",
            ),
            # Persist each step before beginning the next. A running search may
            # repeat after a crash; completed checkpointed steps are retained.
            durability="sync",
        )


async def _run(job_id: UUID) -> None:
    try:
        async with jobs.claim(job_id) as claimed:
            if claimed is None:
                return
            job, connection = claimed
            try:
                from juena_core.server.agent.registry import get_agent, get_agent_resources

                await get_agent("juena")
                build = get_agent_resources("juena").background_specialists[job.specialist]
                runnable = build(checkpointer=AsyncPostgresSaver(connection))["runnable"]
                result = await _invoke(runnable, job)
                package = _final_package(result)
                if _is_result(package):
                    await jobs.finish(
                        connection, job_id,
                        status="succeeded",
                        report=package.text.rstrip(),
                        findings=_written_findings(result, job.run_input["files"]),
                        base_findings=fingerprint(job.run_input["files"]),
                    )
                else:
                    await jobs.finish(
                        connection, job_id,
                        status="failed",
                        report=package.text.rstrip() if package else "",
                        error=NO_REPORT,
                    )
            except TimeoutError:
                await jobs.finish(connection, job_id, status="failed", error=TIMEOUT_MESSAGE)
            except Exception as error:
                logger.exception("Research job %s failed", job_id)
                await jobs.finish(connection, job_id, status="failed", error=str(error))
    except Exception:
        # If the database connection was lost, neither checkpoints nor finish
        # can write through it. Leave the job available for the next poll.
        logger.exception("Research job %s could not persist; leaving it for recovery", job_id)
    # CancelledError deliberately propagates: closing claim releases ownership
    # without changing the durable status or discarding the checkpoint.


async def _poll() -> None:
    while True:
        try:
            for job_id in await jobs.active_ids():
                if job_id not in _running:
                    task = asyncio.create_task(
                        _run(job_id), name=f"research-{job_id}",
                        context=contextvars.Context(),
                    )
                    _running[job_id] = task
                    task.add_done_callback(lambda _t, key=job_id: _running.pop(key, None))
        except Exception:
            logger.exception("Unable to poll background research")
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """One database-driven execution path for both new and interrupted jobs."""
    poller = asyncio.create_task(_poll(), name="research-poller", context=contextvars.Context())
    try:
        yield
    finally:
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)
        tasks = list(_running.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _running.clear()
