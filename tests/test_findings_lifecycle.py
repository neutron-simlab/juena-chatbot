"""The `/findings/` hand-off contract: what crosses, what merges, what conflicts.

Each test below stands for a way the serial happy path -- one specialist writes,
the next one reads -- stops being the only case: a second delegation carrying the
supervisor's private state, a background job returning a snapshot it never
touched, two writers reaching the same path, or a crash on either side of the
checkpoint that records delivery.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from juena.agents.delegation import SpecialistDelegate, with_delegation_boundary
from juena.agents.findings import (
    conflict_path,
    findings_delta,
    fingerprint,
    validate_findings,
)
from juena.research.delivery import describe, plan_merge

JOB = UUID("9ac5d3f2-0000-4000-8000-000000000001")


def _file(content: str) -> dict:
    return {"content": content, "encoding": "utf-8"}


class _Recorder(Runnable):
    """Stands in for a compiled specialist and records what it was handed."""

    def __init__(self, produces: dict | None = None) -> None:
        self.received: dict | None = None
        self._produces = produces or {}

    def invoke(self, input, config=None, **kwargs):  # noqa: A002, ANN001, ANN003
        self.received = input
        return {
            "messages": [AIMessage(content="<specialist_report>…</specialist_report>")],
            "files": {**input.get("files", {}), **self._produces},
            "sandbox_execution_events": [{"leaked": True}],
        }

    async def ainvoke(self, input, config=None, **kwargs):  # noqa: A002, ANN001, ANN003
        return self.invoke(input, config, **kwargs)


# --------------------------------------------------------------------------- #
# What crosses into a specialist
# --------------------------------------------------------------------------- #


def test_the_supervisors_private_working_memory_does_not_reach_a_specialist() -> None:
    """A specialist is told it cannot see the conversation. It could.

    `SubAgentMiddleware` copies the parent's whole state in, and the supervisor
    keeps evicted conversation turns under `/conversation_history/` in the same
    `files` channel that carries findings. So the part of the conversation the
    supervisor had already thrown away was being handed to the one agent whose
    prompt promises it has no access to any of it.
    """
    inner = _Recorder()
    delegate = SpecialistDelegate(inner, name="software-specialist")

    delegate.invoke(
        {
            "messages": [HumanMessage("summarise the fitting code")],
            "files": {
                "/findings/earlier.md": _file("what the last specialist found"),
                "/inputs/uploaded.csv": _file("q,I\n0.1,42"),
                "/conversation_history/thread-1.md": _file("the evicted transcript"),
                "/memories/AGENTS.md": _file("the user's saved preferences"),
                "/scratch.txt": _file("an unmatched top-level path"),
            },
        }
    )

    assert set(inner.received["files"]) == {
        "/findings/earlier.md",
        "/inputs/uploaded.csv",
    }


def test_a_state_field_nobody_allowlisted_does_not_cross_either() -> None:
    """Built from scratch rather than filtered, so new keys are absent by default.

    A denylist would have to be revisited on every upstream release that adds a
    state field; this way the next one simply does not cross until someone
    decides it should.
    """
    inner = _Recorder()

    SpecialistDelegate(inner, name="x").invoke(
        {
            "messages": [HumanMessage("go")],
            "files": {},
            "delivered_research_job_ids": ["a-job-id"],
            "some_future_field": {"invented": "later"},
        }
    )

    assert set(inner.received) == {"messages", "files"}


def test_only_the_report_and_new_findings_come_back() -> None:
    """The return trip is an allowlist too.

    Deep Agents copies the subagent's whole state back except `messages`, so a
    scratch file, a sandbox event, or a finding the run merely inherited all
    reach the conversation.
    """
    inner = _Recorder(
        produces={
            "/findings/new.md": _file("what this run established"),
            "/workspace/scratch.py": _file("disposable"),
        }
    )

    result = SpecialistDelegate(inner, name="x").invoke(
        {
            "messages": [HumanMessage("go")],
            "files": {"/findings/earlier.md": _file("inherited, untouched")},
        }
    )

    assert set(result["files"]) == {"/findings/new.md"}
    assert "sandbox_execution_events" not in result
    assert result["messages"][-1].text.startswith("<specialist_report>")


def test_registration_wraps_every_specialist() -> None:
    wrapped = with_delegation_boundary(
        [{"name": "a", "description": "d", "runnable": _Recorder()}]
    )

    assert isinstance(wrapped[0]["runnable"], SpecialistDelegate)
    assert wrapped[0]["name"] == "a"


# --------------------------------------------------------------------------- #
# What a background job returns
# --------------------------------------------------------------------------- #


def test_a_job_that_changed_nothing_returns_nothing() -> None:
    seed = {"/findings/survey.md": _file("unchanged")}

    assert findings_delta(seed, dict(seed)) == {}


def test_an_untouched_seeded_finding_never_overwrites_a_newer_value() -> None:
    """The delta is what stops a stale copy reverting a live edit.

    The job read `survey.md`, never touched it, and finished after the
    conversation had edited it. A prefix filter would return the job's stale copy
    and collection would write it straight over the newer one.
    """
    seed = {"/findings/survey.md": _file("as the job found it")}
    result = {"files": dict(seed)}

    delta = findings_delta(seed, result["files"])
    merged, conflicts = plan_merge(
        findings=delta,
        base_findings=fingerprint(seed),
        current={"/findings/survey.md": _file("edited while the job ran")},
        job_id=JOB,
    )

    assert delta == {}
    assert merged == {}
    assert conflicts == []


# --------------------------------------------------------------------------- #
# Merging a finished job
# --------------------------------------------------------------------------- #


def test_an_unchanged_base_merges_normally() -> None:
    seed = {"/findings/survey.md": _file("original")}
    merged, conflicts = plan_merge(
        findings={"/findings/survey.md": _file("original\nplus new evidence")},
        base_findings=fingerprint(seed),
        current={"/findings/survey.md": _file("original")},
        job_id=JOB,
    )

    assert merged == {"/findings/survey.md": _file("original\nplus new evidence")}
    assert conflicts == []


def test_a_changed_base_preserves_the_conversation_and_parks_the_job() -> None:
    """Never last-writer-wins: two people wrote, and both versions survive.

    The conversation's copy stays canonical at its own path; the job's goes to
    the reserved namespace, which names the job it came from.
    """
    merged, conflicts = plan_merge(
        findings={"/findings/survey.md": _file("what the job concluded")},
        base_findings={"/findings/survey.md": "original"},
        current={"/findings/survey.md": _file("what a specialist concluded meanwhile")},
        job_id=JOB,
    )
    reserved = conflict_path("/findings/survey.md", JOB)

    assert "/findings/survey.md" not in merged, "the conversation's version was overwritten"
    assert merged == {reserved: _file("what the job concluded")}
    assert conflicts == [("/findings/survey.md", reserved)]
    assert reserved == f"/findings/_conflicts/{JOB}/survey.md"


def test_a_path_the_job_never_read_but_the_conversation_created_is_a_conflict() -> None:
    """Absent from the base is not the same as matching an empty base."""

    merged, conflicts = plan_merge(
        findings={"/findings/survey.md": _file("the job's version")},
        base_findings={},
        current={"/findings/survey.md": _file("created while the job ran")},
        job_id=JOB,
    )

    assert conflicts and merged == {conflict_path("/findings/survey.md", JOB): _file("the job's version")}


def test_the_conflict_path_is_deterministic_so_retrying_is_safe() -> None:
    """Collection may run twice. Twice must not mean two files."""

    first = plan_merge(
        findings={"/findings/survey.md": _file("v")},
        base_findings={"/findings/survey.md": "base"},
        current={"/findings/survey.md": _file("moved on")},
        job_id=JOB,
    )
    second = plan_merge(
        findings={"/findings/survey.md": _file("v")},
        base_findings={"/findings/survey.md": "base"},
        current={"/findings/survey.md": _file("moved on")},
        job_id=JOB,
    )

    assert first == second


def test_the_conflict_report_names_paths_without_inlining_them() -> None:
    """Two copies of a document in message history is what we just removed."""

    text = describe(
        "<specialist_report>…</specialist_report>",
        [("/findings/survey.md", conflict_path("/findings/survey.md", JOB))],
        JOB,
    )

    assert "/findings/survey.md" in text
    assert str(JOB) in text
    assert "what the job concluded" not in text


# --------------------------------------------------------------------------- #
# The stored shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "stored",
    [
        {"/findings/a.md": {"content": "text", "encoding": "utf-8"}},
        {"/findings/a.md": {"content": "text"}},
    ],
)
def test_a_round_trip_keeps_the_file_data_shape(stored: dict) -> None:
    valid = validate_findings(stored)

    assert valid["/findings/a.md"]["content"] == "text"
    assert valid["/findings/a.md"]["encoding"] == "utf-8"


@pytest.mark.parametrize(
    "broken",
    [
        {"/findings/a.md": "a bare string, not FileData"},
        {"/findings/a.md": {"content": None}},
        {"/findings/a.md": {"no_content": "at all"}},
        "not a mapping at all",
        None,
    ],
)
def test_a_malformed_row_is_dropped_rather_than_merged(broken) -> None:  # noqa: ANN001
    """The column is JSON; nothing on the database side guarantees the shape.

    A malformed entry merged into `files` breaks every later read of that path,
    not only this one.
    """
    assert validate_findings(broken) == {}


def test_a_reserved_conflict_path_cannot_arrive_from_a_job() -> None:
    """The backend refuses the write, and validation refuses the stored row.

    Only the collection boundary may fill `_conflicts/`. A job row naming a path
    inside it would otherwise let a background run choose where the conversation
    believes its own conflicting version lives.
    """
    merged, conflicts = plan_merge(
        findings={conflict_path("/findings/survey.md", uuid4()): _file("forged")},
        base_findings={},
        current={},
        job_id=JOB,
    )

    assert merged == {} and conflicts == []


@pytest.mark.parametrize(
    "path",
    [
        "/conversation_history/summary.md",
        "/inputs/data.dat",
        "/workspace/outputs/plot.png",
        "notes.md",
        "/findings_elsewhere/x.md",
    ],
)
def test_a_stored_path_outside_the_hand_off_channel_is_dropped(path: str) -> None:
    """Validation is the namespace boundary, not only the shape boundary.

    The column is JSON and a background model authored what went into it. A row
    reading `/conversation_history/summary.md` would, on merge, overwrite the
    supervisor's own working memory -- where summarization parks evicted turns --
    with text nothing verified.
    """
    assert validate_findings({path: _file("anything")}) == {}

    merged, _ = plan_merge(
        findings={path: _file("anything")},
        base_findings={},
        current={},
        job_id=JOB,
    )
    assert merged == {}


# --------------------------------------------------------------------------- #
# Acknowledging delivery exactly once
# --------------------------------------------------------------------------- #


def _runtime(state: dict) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        state=state,
        context=SimpleNamespace(user_id=str(uuid4()), thread_id="t1"),
        tool_call_id="call-1",
    )


def _finished_job(user_id) -> object:  # noqa: ANN001
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    return SimpleNamespace(
        id=JOB,
        user_id=user_id,
        specialist="software-specialist",
        objective="survey the fitting code",
        status="succeeded",
        report="<specialist_report>…</specialist_report>",
        findings={"/findings/survey.md": _file("what the job found")},
        base_findings={},
        error="",
        created_at=now,
        finished_at=now,
    )


@pytest.mark.asyncio
async def test_a_crash_before_the_checkpoint_leaves_the_job_collectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delivery is acknowledged by the write that performs it, not before it.

    `collected_at` used to be set inside the tool. If anything failed between
    that database write and the graph checkpoint, the row said delivered and the
    conversation never saw the report -- a silent, unrecoverable loss.
    """
    from juena.research import jobs
    from juena.tools.research import build_research_tools

    collected: list = []
    runtime = _runtime({"files": {}, "delivered_research_job_ids": []})
    job = _finished_job(UUID(runtime.context.user_id))

    monkeypatch.setattr(jobs, "get", lambda *_a, **_k: _async(job))
    monkeypatch.setattr(jobs, "mark_collected", lambda job_id: _async(collected.append(job_id)))

    tools = {tool.name: tool for tool in build_research_tools(["software-specialist"])}
    command = await tools["check_research"].coroutine(job_id=str(JOB), runtime=runtime)

    assert collected == [], "the row was marked before the result was durable"
    assert command.update["delivered_research_job_ids"] == [str(JOB)]
    assert "/findings/survey.md" in command.update["files"]


@pytest.mark.asyncio
async def test_a_delivered_job_is_acknowledged_once_and_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker is what the next call reconciles against.

    Not the message history: summarization moves and drops old tool messages, so
    a missing report would read as an undelivered job long after the user saw it.
    """
    from juena.research import jobs
    from juena.tools.research import build_research_tools

    collected: list = []
    runtime = _runtime({"files": {}, "delivered_research_job_ids": [str(JOB)]})
    job = _finished_job(UUID(runtime.context.user_id))

    monkeypatch.setattr(jobs, "get", lambda *_a, **_k: _async(job))
    monkeypatch.setattr(jobs, "mark_collected", lambda job_id: _async(collected.append(job_id)))

    tools = {tool.name: tool for tool in build_research_tools(["software-specialist"])}
    answer = await tools["check_research"].coroutine(job_id=str(JOB), runtime=runtime)

    assert collected == [JOB], "the row was never reconciled against the marker"
    assert isinstance(answer, str)
    assert "already reported" in answer


async def _async(value):  # noqa: ANN001, ANN202
    return value
