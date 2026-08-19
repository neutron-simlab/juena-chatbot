"""Run a specialist detached from the request that asked for it.

deepagents ships async subagents for this, and they need an Agent Protocol
server -- a `langgraph.json`, `langgraph-api`, the LangGraph Platform deployment
shape. JüNA has none of that, and does not need it: a specialist is a compiled
graph this process built and holds, so backgrounding it is an `asyncio` task
plus a row in Postgres.

What a background specialist gives up is the ability to ask. It is compiled
without `ask_user` and without the sandbox, so no `interrupt()` can be raised
with nobody on the other end -- which is also why the run needs no checkpointer
and has nothing to resume.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage

from juena.agents.backends import FINDINGS_PREFIX
from juena.core.log import get_logger
from juena.research import jobs
from juena.research.constants import DEFAULT_TIMEOUT_SECONDS, TIMEOUT_MESSAGE
from juena.server.agent.runtime_model_middleware import RuntimeModelContext

logger = get_logger(__name__)

#: Paths a background specialist is seeded with. `/findings/` is what earlier
#: specialists established and `/inputs/` is what the user staged; everything
#: else in the conversation's state is the supervisor's own working memory.
SEEDED_PREFIXES = (FINDINGS_PREFIX, "/inputs/")

NO_REPORT = (
    "The specialist finished without producing a report. Nothing it did mid-run "
    "can be treated as a result."
)

#: Tasks are held here for their lifetime. `asyncio.create_task` keeps only a
#: weak reference, so a task nobody holds can be garbage-collected mid-run --
#: silently, with the row left saying `running` forever.
_running: dict[UUID, asyncio.Task] = {}


def active_count() -> int:
    """How many jobs this process is currently running. For tests and shutdown."""
    return len(_running)


def _seed_files(state_values: dict[str, Any]) -> dict[str, Any]:
    """The parent thread's findings and staged inputs, and nothing else.

    A subagent invoked through `task` inherits the parent's whole `files` state
    for free. This one is invoked standalone, so what it starts with is decided
    here rather than by LangGraph.
    """
    files = state_values.get("files") or {}
    return {
        path: data
        for path, data in files.items()
        if path.startswith(SEEDED_PREFIXES)
    }


def _report_text(result: dict[str, Any]) -> str:
    """The specialist's final package: the last AIMessage carrying text.

    `SpecialistOutcomeMiddleware` composes that message from the structured
    report and the server's own verification, which is exactly what the `task`
    tool forwards to the supervisor. Reading it the same way keeps a collected
    background result indistinguishable from a synchronous one.
    """
    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage):
            text = message.text.rstrip() if message.text else ""
            if text:
                return text
    return NO_REPORT


def _written_findings(result: dict[str, Any]) -> dict[str, Any]:
    """Files the run left under `/findings/`, to hand back on collection."""
    files = result.get("files") or {}
    return {
        path: data for path, data in files.items() if path.startswith(FINDINGS_PREFIX)
    }


async def _invoke(
    runnable: Any,
    *,
    job_id: UUID,
    objective: str,
    seed: dict[str, Any],
    thread_id: str,
    user_id: UUID,
    provider: str,
    model: str,
) -> dict[str, Any]:
    return await runnable.ainvoke(
        {
            "messages": [HumanMessage(content=objective)],
            "files": seed,
        },
        # Its own graph thread, so nothing writes into the conversation's
        # checkpoint while the user is still using it. The *context* thread stays
        # the chat's, because that is the identity the artifact store and the
        # memory namespace resolve against.
        config={"configurable": {"thread_id": f"research:{job_id}"}},
        context=RuntimeModelContext(
            provider=provider,
            model=model,
            thread_id=thread_id,
            user_id=str(user_id),
        ),
    )


async def _run(
    job_id: UUID,
    objective: str,
    thread_id: str,
    user_id: UUID,
    specialist: str,
    provider: str,
    model: str,
    timeout_seconds: int,
) -> None:
    """One job, from `queued` to a terminal row. Never raises."""
    try:
        from juena.server.agent.registry import get_agent, get_agent_resources

        await jobs.mark_running(job_id)

        agent = await get_agent("juena")
        resources = get_agent_resources("juena")
        runnable = resources.background_specialists.get(specialist)
        if runnable is None:
            await jobs.finish(
                job_id,
                status="failed",
                error=f"No background specialist named '{specialist}' is registered",
            )
            return

        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        seed = _seed_files(snapshot.values if snapshot else {})

        result = await asyncio.wait_for(
            _invoke(
                runnable,
                job_id=job_id,
                objective=objective,
                seed=seed,
                thread_id=thread_id,
                user_id=user_id,
                provider=provider,
                model=model,
            ),
            timeout=timeout_seconds,
        )
        await jobs.finish(
            job_id,
            status="succeeded",
            report=_report_text(result),
            findings=_written_findings(result),
        )
        logger.info("Research job %s finished (%s)", job_id, specialist)

    except asyncio.TimeoutError:
        logger.warning("Research job %s timed out after %ds", job_id, timeout_seconds)
        await jobs.finish(job_id, status="failed", error=TIMEOUT_MESSAGE)
    except asyncio.CancelledError:
        # Shutdown, or an explicit cancel. The row is written by whoever
        # cancelled; re-raise so the event loop sees a proper cancellation.
        raise
    except Exception as error:
        logger.exception("Research job %s failed", job_id)
        await jobs.finish(job_id, status="failed", error=str(error))


def launch(
    *,
    job_id: UUID,
    objective: str,
    thread_id: str,
    user_id: UUID,
    specialist: str,
    provider: str,
    model: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Start the job and return immediately.

    The task belongs to the event loop, not to the request that created it, so
    the specialist keeps working after the SSE response has closed.
    """
    task = asyncio.create_task(
        _run(
            job_id,
            objective,
            thread_id,
            user_id,
            specialist,
            provider,
            model,
            timeout_seconds,
        ),
        name=f"research-{job_id}",
    )
    _running[job_id] = task
    task.add_done_callback(lambda _t: _running.pop(job_id, None))


async def shutdown() -> None:
    """Cancel every running job and record why, on the way down."""
    if not _running:
        return

    logger.info("Cancelling %d in-flight research job(s)", len(_running))
    for task in list(_running.values()):
        task.cancel()
    await asyncio.gather(*list(_running.values()), return_exceptions=True)
    # The rows are failed here rather than in each task: a cancelled task cannot
    # be relied on to finish its own `await`.
    await jobs.fail_orphans()
    _running.clear()
