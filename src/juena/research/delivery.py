"""Handing a finished background job to the conversation, exactly once.

Two failures sit either side of the same moment. Marking the job collected before
its result is checkpointed loses the report if anything fails in between -- the
row says delivered and the conversation never saw it. Marking it afterwards has
no "afterwards" to hook: a tool returns before the graph commits.

So delivery is acknowledged by the same atomic write that performs it. The
`Command` that carries the report and the findings also stamps the job id into a
private, checkpointed state field, and the *next* call reconciles the database
against that stamp:

* stamp present -> the conversation has it; mark the row collected and say so;
* stamp absent  -> the write never landed; hand the identical `Command` back.

Both windows are covered, and replaying is safe because the conflict paths are
deterministic. The marker is deliberately not inferred from message history:
summarization moves and drops old tool messages, so an absent report would read
as an undelivered job long after the user had seen it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired
from uuid import UUID

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr

from juena.agents.findings import (
    FindingsMap,
    conflict_path,
    is_conflict_path,
    validate_findings,
)

#: Absent from `base_findings` is not the same as empty: a path the job never
#: read, which the conversation has since created, is a conflict rather than a
#: match against "".
_UNSEEN = object()

DELIVERED_KEY = "delivered_research_job_ids"


class ResearchDeliveryState(AgentState):
    """The durable record of which jobs this conversation has already received."""

    delivered_research_job_ids: NotRequired[
        Annotated[list[str], operator.add, PrivateStateAttr]
    ]


class ResearchDeliveryMiddleware(AgentMiddleware[ResearchDeliveryState]):
    """Declares the delivery marker. Private, so it never crosses to a specialist.

    No hooks: the field exists so `check_research` can write it inside the same
    `Command` that delivers a report, and read it back on a later turn.
    """

    state_schema = ResearchDeliveryState


def delivered_ids(state: Any) -> set[str]:
    """Job ids this conversation has durably received."""
    values = state if isinstance(state, dict) else {}
    return {str(item) for item in (values.get(DELIVERED_KEY) or [])}


def current_findings(state: Any) -> dict[str, Any]:
    """The conversation's findings as they stand now."""
    values = state if isinstance(state, dict) else {}
    return dict(values.get("files") or {})


def plan_merge(
    *,
    findings: Any,
    base_findings: dict[str, str] | None,
    current: dict[str, Any],
    job_id: UUID,
) -> tuple[FindingsMap, list[tuple[str, str]]]:
    """Split a job's findings into what may merge and what must be set aside.

    A background file may overwrite the conversation's copy only when that copy
    is still the revision the job read. If it changed in the meantime the
    conversation's version is canonical and the job's is preserved under the
    reserved conflict namespace, where a specialist cannot write and the path
    names the job it came from.
    """
    base = base_findings or {}
    merged: FindingsMap = {}
    conflicts: list[tuple[str, str]] = []

    for path, data in validate_findings(findings).items():
        # A run cannot author one of these; refuse to merge it if one appears.
        if is_conflict_path(path):
            continue
        existing = current.get(path)
        if existing is None:
            merged[path] = data
            continue
        unchanged = str((existing or {}).get("content", "")) == base.get(path, _UNSEEN)
        if unchanged:
            merged[path] = data
        else:
            reserved = conflict_path(path, job_id)
            merged[reserved] = data
            conflicts.append((path, reserved))

    return merged, conflicts


def describe(report: str, conflicts: list[tuple[str, str]], job_id: UUID) -> str:
    """The report, plus where anything that could not be merged was put.

    Paths only. The conflicting text is in the conversation's own files for the
    supervisor to read if it turns out to matter -- inlining it here would put
    two versions of the same document into message history for the rest of the
    session, to settle a question that usually does not arise.
    """
    if not conflicts:
        return report
    lines = [
        report,
        "",
        f"<merge_conflicts job=\"{job_id}\">",
        "These findings changed in this conversation while the job was running, so "
        "the conversation's version was kept. The job's version is at the second "
        "path if you need to compare them.",
        *[f"- {original}  ->  {reserved}" for original, reserved in conflicts],
        "</merge_conflicts>",
    ]
    return "\n".join(lines)
