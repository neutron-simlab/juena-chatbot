"""The `/findings/` hand-off: what it holds, what may cross, and what conflicts.

`/findings/` is **conversation-scoped, mutable, specialist-authored** hand-off
data. A model wrote every word of it and the server verified none of it; it is
not an evidence ledger, and `<verified_by_server>` remains the only authority on
what actually ran. Everything here follows from that: findings are compared,
merged, and set aside, never trusted.

The helpers live in one module because four callers need the same answers --
`juena.agents.delegation` deciding what enters a specialist,
`juena.research.runner` deciding what a background job produced,
`juena.tools.research` merging a finished job back, and
`juena.server.database.models` storing it -- and three of them had their own
slightly different idea of the shape before this.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepagents.backends.protocol import FileData

from juena.agents.backends import FINDINGS_PREFIX

#: What the graph's `files` channel and the `research_jobs.findings` column
#: actually hold. Deep Agents stores `FileData`, not plain strings -- the column
#: was annotated `dict[str, str]` for a while and the annotation was simply wrong.
FindingsMap = dict[str, FileData]

#: The only paths that may cross into a specialist. Everything else in the
#: supervisor's `files` channel -- notably `/conversation_history/`, where
#: summarization parks evicted turns -- is the supervisor's own working memory.
CROSSING_PREFIXES: tuple[str, ...] = ("/inputs/", FINDINGS_PREFIX)

#: Where a background result goes when it cannot be merged. Server-owned: a
#: specialist is refused a write here, so anything under it was put there by the
#: collection boundary and names the job it came from.
CONFLICT_PREFIX = f"{FINDINGS_PREFIX}_conflicts/"


def is_conflict_path(path: str) -> bool:
    """Whether a path lies in the reserved conflict namespace."""
    return path.startswith(CONFLICT_PREFIX)


def conflict_path(path: str, job_id: Any) -> str:
    """Where a background version lands when the conversation moved on.

    Deterministic on ``(path, job_id)``, which is what makes collection safe to
    retry: replaying the same job writes the same key rather than accumulating
    ``survey-2.md``, ``survey-3.md``.
    """
    return f"{CONFLICT_PREFIX}{job_id}/{path[len(FINDINGS_PREFIX):]}"


def crossing_files(files: Mapping[str, Any] | None) -> dict[str, Any]:
    """The staged inputs and findings a specialist is allowed to receive."""
    return {
        path: data
        for path, data in (files or {}).items()
        if path.startswith(CROSSING_PREFIXES)
    }


def findings_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> FindingsMap:
    """Findings a run created or changed -- never the ones it merely inherited.

    A background job is seeded with the conversation's findings, so a plain
    prefix filter over its result returns that whole snapshot back. Collecting
    it would rewrite every finding with a copy the job never touched, quietly
    reverting anything edited while the job ran.
    """
    seeded = before or {}
    return {
        path: data
        for path, data in (after or {}).items()
        if path.startswith(FINDINGS_PREFIX) and data != seeded.get(path)
    }


def fingerprint(files: Mapping[str, Any] | None) -> dict[str, str]:
    """The base revision a job read, small enough to keep on the job row.

    Content rather than a hash: findings are short, and storing the text means a
    conflict can be explained without a second lookup.
    """
    return {
        path: str((data or {}).get("content", ""))
        for path, data in (files or {}).items()
        if path.startswith(FINDINGS_PREFIX)
    }


def validate_findings(value: Any) -> FindingsMap:
    """Coerce whatever came back from PostgreSQL into `FileData` shape.

    The column is JSON, so nothing on the database side guarantees the shape or
    the paths -- and this is the one place a row becomes graph state. A malformed
    entry merged into the `files` channel breaks every later read of that path;
    a *misplaced* one is worse, because `/conversation_history/summary.md` in
    that column would overwrite the supervisor's own working memory with text a
    background model wrote.

    So the path is checked as strictly as the shape: `/findings/`, and never the
    reserved conflict namespace, which only the collection boundary may fill.
    Entries failing either check are dropped rather than merged.
    """
    if not isinstance(value, Mapping):
        return {}
    valid: FindingsMap = {}
    for path, data in value.items():
        if not isinstance(path, str) or not isinstance(data, Mapping):
            continue
        if not path.startswith(FINDINGS_PREFIX) or is_conflict_path(path):
            continue
        content = data.get("content")
        if not isinstance(content, str):
            continue
        entry: dict[str, Any] = {
            "content": content,
            "encoding": data.get("encoding") if isinstance(data.get("encoding"), str) else "utf-8",
        }
        for optional in ("created_at", "modified_at"):
            if isinstance(data.get(optional), str):
                entry[optional] = data[optional]
        valid[path] = entry  # type: ignore[assignment]
    return valid
