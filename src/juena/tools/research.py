"""Start, check and list background research from the supervisor.

The counterpart to `task`. Where `task` runs a specialist inside the tool call
and returns its report, these three start one detached and let a later turn
collect it -- so a ten-minute survey no longer holds the conversation, and
closing the browser no longer abandons the work.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from juena_core.log import get_logger
from juena.research import jobs, runner
from juena.research.delivery import (
    current_findings,
    deliver,
    delivered_ids,
    describe,
    plan_merge,
)
from juena.research.constants import BUSY_MESSAGE, DEFAULT_CONCURRENCY, TERMINAL_STATUSES
from juena.server.database.models import ResearchJob

logger = get_logger(__name__)

_OBJECTIVE_PREVIEW = 80

START_DESCRIPTION = """Delegate a long objective to a specialist that runs in the background.

Use this instead of `task` only when the objective is genuinely long -- a survey
spanning several repositories, or a question needing many searches to settle --
**and** the user has something else to talk about while it runs. For everything
else use `task`, which answers in the same turn and lets the user watch it work.

A background specialist cannot ask the user anything and cannot run code. If the
objective might need a clarification or a sandbox execution, use `task`.

The objective must be self-contained, exactly as for `task`: the specialist
cannot see this conversation.

Returns a job id. Do not wait for it -- reply to the user, saying what you
started and that you will report back."""

CHECK_DESCRIPTION = """Read the result of one background research job by its id.

Returns the specialist's full report once it has finished, or its status if it
has not. It is the same package `task` returns: the specialist's own finding,
evidence and limitations, alongside the server's verification.

Whatever the run wrote to `/findings/` is merged into this conversation, so a
specialist you delegate to afterwards reads it the same way it reads a finding
left by any earlier delegation.

A finished job is handed over only once; after that it stops appearing in
`list_research`."""

LIST_DESCRIPTION = """List the background research on this conversation you have not yet reported.

Call this at the start of a turn. Returns each job's id, specialist, status, age
in seconds, and the beginning of its objective. Anything `finished` should be
collected with `check_research` and worked into your answer before you address
whatever the user just asked."""


def _identity(runtime: ToolRuntime) -> tuple[UUID, str] | None:
    """The authenticated user and thread this tool call belongs to.

    Read from runtime context rather than from an argument: a job id the model
    could pair with any user id would be an ownership check the model controls.
    """
    context = getattr(runtime, "context", None)
    user_id = getattr(context, "user_id", None)
    thread_id = getattr(context, "thread_id", None)
    if not user_id or not thread_id:
        return None
    try:
        return UUID(str(user_id)), str(thread_id)
    except ValueError:
        return None


def _age_seconds(job: ResearchJob) -> int:
    end = job.finished_at or datetime.now(UTC)
    return max(int((end - job.created_at).total_seconds()), 0)


def _summary(job: ResearchJob) -> dict[str, Any]:
    objective = job.objective.strip()
    if len(objective) > _OBJECTIVE_PREVIEW:
        objective = objective[:_OBJECTIVE_PREVIEW].rstrip() + "…"
    return {
        "job_id": str(job.id),
        "specialist": job.specialist,
        # One word the model acts on, rather than five it has to interpret.
        # Every terminal status reads as `finished`, cancellation included: each
        # one has an outcome to collect, and collecting is what clears the job
        # off this list.
        "status": "finished" if job.status in TERMINAL_STATUSES else job.status,
        "age_seconds": _age_seconds(job),
        "objective": objective,
    }


def build_research_tools(
    specialist_names: list[str],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list:
    """Build the three background-research tools for the supervisor."""

    allowed = ", ".join(f"`{name}`" for name in specialist_names)

    @tool(description=f"{START_DESCRIPTION}\n\nAvailable specialists: {allowed}")
    async def start_research(
        specialist: str,
        objective: str,
        runtime: ToolRuntime,
    ) -> str:
        if specialist not in specialist_names:
            return (
                f"There is no specialist called '{specialist}'. "
                f"The available ones are: {allowed}."
            )

        identity = _identity(runtime)
        if identity is None:
            return "Background research is unavailable: this run has no authenticated user."
        user_id, thread_id = identity

        context = runtime.context
        job = await jobs.create(
            user_id=user_id,
            thread_id=thread_id,
            specialist=specialist,
            objective=objective,
            concurrency=concurrency,
        )
        if job is None:
            return BUSY_MESSAGE

        runner.launch(
            job_id=job.id,
            objective=objective,
            thread_id=thread_id,
            user_id=user_id,
            specialist=specialist,
            provider=context.provider,
            model=context.model,
        )
        logger.info("Started research job %s (%s) on thread %s", job.id, specialist, thread_id)
        return (
            f"Started background research job {job.id} with {specialist}. "
            "It usually takes a few minutes. Tell the user it is running and "
            "carry on with the rest of their message -- do not wait for it."
        )

    @tool(description=CHECK_DESCRIPTION)
    async def check_research(job_id: str, runtime: ToolRuntime) -> str | Command:
        identity = _identity(runtime)
        if identity is None:
            return "Background research is unavailable: this run has no authenticated user."
        user_id, _thread_id = identity

        try:
            parsed = UUID(job_id)
        except ValueError:
            return f"'{job_id}' is not a research job id."

        job = await jobs.get(parsed, user_id)
        if job is None:
            return f"No research job {job_id} belongs to this conversation."

        if job.status not in TERMINAL_STATUSES:
            return (
                f"Research job {job_id} ({job.specialist}) is still {job.status}, "
                f"{_age_seconds(job)}s in. Tell the user it is not ready yet rather "
                "than checking again in this turn."
            )
        # Asked before the outcome is read, because every terminal status is
        # delivered through the marker -- a failure reported twice is as wrong as
        # a report delivered twice.
        if str(job.id) in delivered_ids(runtime.state):
            # The outcome reached a checkpoint; the row simply never learned.
            # Mark it now rather than handing the same result over again.
            await jobs.mark_collected(job.id)
            return (
                f"Research job {job_id} was already reported in this conversation. "
                "Use what is already here rather than repeating it."
            )

        if job.status == "cancelled":
            return deliver(
                text=(
                    f"Research job {job_id} ({job.specialist}) was cancelled and "
                    "produced no result. Tell the user, and ask whether they still "
                    "want it before starting it again."
                ),
                job_id=job.id,
                tool_call_id=runtime.tool_call_id,
            )
        if job.status == "failed":
            return deliver(
                text=(
                    f"Research job {job_id} ({job.specialist}) failed: {job.error}. "
                    "Tell the user plainly, and offer to run it again with `task` so "
                    "they can watch it and answer any question it raises."
                ),
                job_id=job.id,
                tool_call_id=runtime.tool_call_id,
            )

        merged, conflicts = plan_merge(
            findings=job.findings,
            base_findings=job.base_findings,
            current=current_findings(runtime.state),
            job_id=job.id,
        )
        return deliver(
            text=describe(job.report, conflicts, job.id),
            job_id=job.id,
            tool_call_id=runtime.tool_call_id,
            files=merged,
        )

    @tool(description=LIST_DESCRIPTION)
    async def list_research(runtime: ToolRuntime) -> str:
        identity = _identity(runtime)
        if identity is None:
            return "[]"
        user_id, thread_id = identity

        pending = await jobs.for_thread(thread_id, user_id)
        # Reconcile before listing: a job whose report reached a checkpoint but
        # whose row never learned would otherwise be offered again every turn.
        delivered = delivered_ids(runtime.state)
        outstanding = []
        for job in pending:
            if str(job.id) in delivered:
                await jobs.mark_collected(job.id)
                continue
            outstanding.append(job)
        if not outstanding:
            return "[]"
        return json.dumps([_summary(job) for job in outstanding], separators=(",", ":"))

    return [start_research, check_research, list_research]
