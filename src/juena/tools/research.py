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
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from juena.core.log import get_logger
from juena.research import jobs, runner
from juena.research.constants import BUSY_MESSAGE, DEFAULT_CONCURRENCY
from juena.server.database.models import ResearchJob

logger = get_logger(__name__)

_OBJECTIVE_PREVIEW = 80

#: How much of each findings file rides back in the tool result. Generous, since
#: it replaces a file the supervisor cannot open, but bounded: this text stays in
#: checkpointed history for the rest of the conversation.
_FINDINGS_PREVIEW = 4000

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
has not. Whatever the run wrote to `/findings/` comes back inside the result, in
a `<findings_written>` block -- that text is the evidence, and it is the only
copy you will get. Do not try to open those paths yourself; if a follow-up
specialist needs the detail, paste the relevant part into its objective.

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
        "status": "finished" if job.status in ("succeeded", "failed") else job.status,
        "age_seconds": _age_seconds(job),
        "objective": objective,
    }


def _report_with_findings(job: ResearchJob) -> str:
    """The report, followed by whatever the run wrote to `/findings/`.

    Inlined rather than left as a path because the supervisor cannot open one:
    its own backend serves `/memories/` only, and a specialist it delegates to
    may be reading `/findings/` from a sandbox workspace this run never touched.
    Truncated per file -- this lands in checkpointed history for the rest of the
    conversation, and the report is meant to be the summary.
    """
    findings = job.findings or {}
    if not findings:
        return job.report

    sections = [job.report, "", "<findings_written>"]
    for path in sorted(findings):
        content = str((findings[path] or {}).get("content", ""))
        if len(content) > _FINDINGS_PREVIEW:
            content = content[:_FINDINGS_PREVIEW].rstrip() + "\n… (truncated)"
        sections.extend([f"--- {path} ---", content])
    sections.append("</findings_written>")
    return "\n".join(sections)


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

        if job.status in ("queued", "running"):
            return (
                f"Research job {job_id} ({job.specialist}) is still {job.status}, "
                f"{_age_seconds(job)}s in. Tell the user it is not ready yet rather "
                "than checking again in this turn."
            )
        if job.status == "cancelled":
            return f"Research job {job_id} was cancelled and produced no result."
        if job.status == "failed":
            await jobs.mark_collected(job.id)
            return (
                f"Research job {job_id} ({job.specialist}) failed: {job.error}. "
                "Tell the user plainly, and offer to run it again with `task` so "
                "they can watch it and answer any question it raises."
            )

        await jobs.mark_collected(job.id)
        # Two ways back, because neither alone is enough.
        #
        # The state merge makes `/findings/` readable to a later specialist --
        # but only where `/findings/` *is* state. With the sandbox enabled a
        # specialist's default backend is its Podman workspace, and nothing in
        # graph state reaches inside it. So the text goes back too: appended to
        # the report, it is the one form the supervisor can always act on,
        # because it can paste it into the next objective.
        return Command(
            update={
                "files": job.findings or {},
                "messages": [
                    ToolMessage(
                        _report_with_findings(job),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    @tool(description=LIST_DESCRIPTION)
    async def list_research(runtime: ToolRuntime) -> str:
        identity = _identity(runtime)
        if identity is None:
            return "[]"
        user_id, thread_id = identity

        pending = await jobs.for_thread(thread_id, user_id)
        if not pending:
            return "[]"
        return json.dumps([_summary(job) for job in pending], separators=(",", ":"))

    return [start_research, check_research, list_research]
