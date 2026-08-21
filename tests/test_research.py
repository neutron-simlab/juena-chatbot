"""Background research: what the runner extracts, and what unattended removes.

Nothing here touches Postgres or a model provider. The database half lives in
`test_postgres_integration.py`, because concurrency and orphan-sweeping are only
meaningful against a real transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from deepagents.backends import CompositeBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from juena.agents import specialist_runtime
from juena.agents.specialist_outcome import SpecialistOutcomeMiddleware, outcome_verified
from juena.agents.specialists.neutron_research import agent as neutron_agent
from juena.agents.specialists.software import agent as software_agent
from juena.research import runner
from juena.sandbox import runtime as sandbox_runtime
from juena.tools.research import _summary, build_research_tools


# --------------------------------------------------------------------------- #
# What crosses into a background run, and what comes back
# --------------------------------------------------------------------------- #


def test_only_findings_and_staged_inputs_are_seeded() -> None:
    """A `task` subagent inherits the parent's whole `files`; this one does not.

    The supervisor's own scratch files are working memory for the conversation,
    not evidence, and sending them would put the specialist's context back where
    delegation exists to keep it from.
    """
    seeded = runner._seed_files(
        {
            "files": {
                "/findings/fitting.md": {"content": "what an earlier run found"},
                "/inputs/current_message.txt": {"content": "the staged message"},
                "/conversation_history/thread-1.md": {"content": "evicted turns"},
                "/memories/AGENTS.md": {"content": "user preferences"},
                "/scratch.md": {"content": "supervisor scribble"},
            }
        }
    )

    assert set(seeded) == {"/findings/fitting.md", "/inputs/current_message.txt"}


def test_seeding_survives_a_thread_with_no_files() -> None:
    assert runner._seed_files({}) == {}
    assert runner._seed_files({"files": None}) == {}


def test_the_report_is_the_last_ai_message_with_text() -> None:
    """The same rule `task` uses, so a collected result reads identically.

    A trailing empty AIMessage after a final tool call is normal and must not
    become an empty report.
    """
    result = {
        "messages": [
            HumanMessage(content="the objective"),
            AIMessage(content="<specialist_report>\nStatus: OK\n</specialist_report>"),
            ToolMessage(content="a tool result", tool_call_id="t1"),
            AIMessage(content=""),
        ]
    }

    assert "Status: OK" in runner._final_package(result).text


def test_a_run_with_no_message_at_all_has_no_package() -> None:
    assert runner._final_package({"messages": []}) is None


def test_only_findings_come_back() -> None:
    """Scratch files a background run made are its own business.

    `check_research` merges whatever this returns straight into supervisor state,
    so anything here becomes visible to the conversation.
    """
    written = runner._written_findings(
        {
            "files": {
                "/findings/sasview-fitting.md": {"content": "paths and line numbers"},
                "/workspace/notes.txt": {"content": "disposable"},
                "/inputs/current_message.txt": {"content": "the objective, echoed"},
            }
        },
        {},
    )

    assert set(written) == {"/findings/sasview-fitting.md"}


def test_a_finding_the_run_only_inherited_does_not_come_back() -> None:
    """The seeded snapshot is not a result, and returning it is not harmless.

    A job starts with the conversation's findings in its state, so a plain prefix
    filter hands every one of them back. Collecting that rewrites each path with
    a copy the run never touched -- reverting anything edited while it worked.
    """
    seed = {
        "/findings/untouched.md": {"content": "written before the job started"},
        "/findings/extended.md": {"content": "original"},
    }
    result = {
        "files": {
            **seed,
            "/findings/extended.md": {"content": "original\nplus what the job found"},
            "/findings/new.md": {"content": "only this run made it"},
        }
    }

    written = runner._written_findings(result, seed)

    assert set(written) == {"/findings/extended.md", "/findings/new.md"}


# --------------------------------------------------------------------------- #
# Unattended removes capability, not prompt
# --------------------------------------------------------------------------- #


def _compiled_kwargs(module: Any, monkeypatch: pytest.MonkeyPatch, build, **kwargs) -> dict:
    """Compile a specialist with `create_agent` stubbed, and return its kwargs."""
    captured: dict[str, Any] = {}

    def fake_create_agent(**agent_kwargs: Any) -> Any:
        captured.update(agent_kwargs)
        return object()

    monkeypatch.setattr(module, "build_chat_model", lambda **_k: object())
    monkeypatch.setattr(module, "create_agent", fake_create_agent)
    # The summarizer is the one middleware that type-checks its model, and a
    # real one would mean a real provider. Nothing here asserts on middleware.
    monkeypatch.setattr(
        specialist_runtime, "create_summarization_middleware", lambda *_a: object()
    )
    build(**kwargs)
    return captured


def _software(monkeypatch: pytest.MonkeyPatch, *, unattended: bool) -> dict:
    return _compiled_kwargs(
        software_agent,
        monkeypatch,
        software_agent.build_software_specialist,
        repo_manager=SimpleNamespace(cache_dir=None),
        vector_index=object(),
        has_repo_research=False,
        context7_tools=(),
        summarizer_model=object(),
        fallback_models=[],
        unattended=unattended,
    )


def _neutron(monkeypatch: pytest.MonkeyPatch, *, unattended: bool) -> dict:
    return _compiled_kwargs(
        neutron_agent,
        monkeypatch,
        neutron_agent.build_neutron_research_specialist,
        tavily_tool=None,
        summarizer_model=object(),
        fallback_models=[],
        unattended=unattended,
    )


@pytest.mark.parametrize("compile_specialist", [_software, _neutron])
def test_an_unattended_specialist_cannot_ask(
    compile_specialist, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ask_user` raises an interrupt, and a background job has nobody to answer.

    Removed rather than discouraged: a prompt that says "do not ask" is a request,
    and a job that asks anyway would hang until its timeout.
    """
    interactive = {tool.name for tool in compile_specialist(monkeypatch, unattended=False)["tools"]}
    background = {tool.name for tool in compile_specialist(monkeypatch, unattended=True)["tools"]}

    assert "ask_user" in interactive
    assert "ask_user" not in background
    # Everything else it could do, it can still do. The flag gates capability
    # that needs a human, not the specialist's expertise.
    assert interactive - background == {"ask_user"}


def test_an_unattended_specialist_is_told_it_is_unattended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background = _software(monkeypatch, unattended=True)["system_prompt"]
    interactive = _software(monkeypatch, unattended=False)["system_prompt"]

    assert "running unattended" in background.lower()
    assert "limitations" in background
    assert "running unattended" not in interactive.lower()


def test_an_unattended_software_specialist_never_gets_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approval and execution are removed together, never one without the other.

    Dropping only the approval middleware would leave `execute` mounted and
    unguarded, which is worse than either alternative.
    """
    monkeypatch.setattr(software_agent, "sandbox_enabled", lambda: True)

    def fail_if_called(**_kwargs: Any) -> Any:
        raise AssertionError("an unattended specialist must not mount the sandbox backend")

    monkeypatch.setattr(software_agent, "build_sandbox_backend", fail_if_called)
    kwargs = _software(monkeypatch, unattended=True)

    assert "execute" not in {tool.name for tool in kwargs["tools"]}
    assert "Sandboxed execution is unavailable" in kwargs["system_prompt"]


class _RefusingSandbox:
    """Stands in for the Podman backend and fails loudly if anything reaches it.

    The container's root filesystem is read-only, so a `/findings/` write that
    got this far used to fail with `Errno 30: Read-only file system`.
    """

    def __getattr__(self, name: str):
        def refuse(*_args: Any, **_kwargs: Any):
            raise AssertionError(f"/findings must never reach the sandbox (called {name})")

        return refuse


def _findings_backends(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """The specialist backend in both shapes: with a sandbox and without."""
    from juena.agents.specialist_runtime import build_specialist_backend

    monkeypatch.setattr(sandbox_runtime.global_config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(
        sandbox_runtime, "RuntimePodmanSandboxBackend", lambda: _RefusingSandbox()
    )
    return [
        ("plain", build_specialist_backend()),
        (
            "sandboxed",
            sandbox_runtime.build_sandbox_backend(
                repo_cache_root=None, skills_dir=None, has_skills=False
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_findings_are_writable_and_readable_with_or_without_a_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/findings/` is graph state in both shapes, under its own full path.

    Three bugs meet here. Left to the backend's default, `/findings/` was the
    Podman container when the sandbox was on -- and that root is read-only, so
    `write_file` failed outright with `Errno 30`. Routing it to a plain
    `StateBackend` instead fixed the write but rewrote the key: `CompositeBackend`
    strips a matched prefix, so `/findings/survey.md` was stored as `/survey.md`,
    indistinguishable from any other top-level file and missed by the filter
    `juena.research` uses. `FindingsStateBackend` does both halves.

    Driven with `ainvoke` deliberately, because that is what the server does and
    the third bug lived only there: `BackendProtocol` implements every `a*`
    method as `asyncio.to_thread(self.<sync>, ...)`, so an `a*` override that
    mapped the path applied it twice. An earlier version of this test called
    `invoke`, passed, and proved nothing about the running system.
    """
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    def call(name: str, args: dict, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )

    path = "/findings/jscatter-example-plot.md"
    for label, backend in _findings_backends(monkeypatch):
        agent = create_agent(
            model=ToolModel(
                responses=[
                    call("write_file", {"file_path": path, "content": "headless is required"}, "w"),
                    call("ls", {"path": "/findings"}, "ls"),
                    call("read_file", {"file_path": path}, "r"),
                    call("grep", {"pattern": "headless", "path": "/findings"}, "g"),
                    call("edit_file", {"file_path": path, "old_string": "required", "new_string": "mandatory"}, "e"),
                    AIMessage(content="done"),
                ]
            ),
            tools=[],
            middleware=[FilesystemMiddleware(backend=backend)],
            checkpointer=InMemorySaver(),
        )
        result = await agent.ainvoke(
            {"messages": [("user", "record a finding")]},
            config={"configurable": {"thread_id": f"findings-{label}"}},
        )
        said = {
            message.tool_call_id: str(message.content)
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }

        assert "Errno 30" not in said["w"], f"{label} could not write a finding"
        assert path in said["ls"], f"{label} cannot list its own finding"
        assert "headless" in said["r"], f"{label} cannot read its own finding"
        # Double-prefixing is the failure mode when the wrong method is overridden:
        # `StateBackend.grep` never routes through `grep_raw`.
        assert path in said["g"] and "/findings/findings/" not in said["g"]
        assert "Successfully replaced" in said["e"], f"{label} cannot extend a finding"
        assert list(result.get("files") or {}) == [path], (
            f"{label} stored the finding under a key nothing else will match"
        )


def test_a_finding_written_by_one_delegation_reaches_the_next() -> None:
    """The claim `/findings/` exists to make true, finally asserted.

    A specialist cannot see the conversation or any earlier delegation; the
    `files` channel is the only thing that crosses. deepagents copies a
    subagent's whole state back except `messages`, `todos` and
    `structured_response`, so `files` rides along -- **but only if the parent has
    a `files` channel at all**, which it gets from `FilesystemMiddleware`. Drop
    that middleware from the supervisor and every finding is silently discarded.

    The reader proves it saw the file by editing it: `edit_file` fails on a path
    that is not there, so the changed content in the parent's state is the proof.
    """
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from juena.agents.backends import build_supervisor_backend
    from juena.agents.specialist_runtime import build_specialist_backend

    path = "/findings/jscatter-example-plot.md"

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    def call(name: str, args: dict, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )

    def specialist(responses: list) -> Any:
        return create_agent(
            model=ToolModel(responses=responses),
            tools=[],
            middleware=[FilesystemMiddleware(backend=build_specialist_backend())],
        )

    writer = specialist([
        call("write_file", {"file_path": path, "content": "## Key Patterns\nheadless first\n"}, "w"),
        AIMessage(content="wrote the finding"),
    ])
    reader = specialist([
        call("ls", {"path": "/findings"}, "ls"),
        call("edit_file", {"file_path": path, "old_string": "headless first", "new_string": "READ BY THE SECOND SPECIALIST"}, "e"),
        AIMessage(content="read the finding"),
    ])

    supervisor = create_agent(
        model=ToolModel(
            responses=[
                call("task", {"description": "record it", "subagent_type": "writer"}, "t1"),
                call("task", {"description": "build on it", "subagent_type": "reader"}, "t2"),
                AIMessage(content="done"),
            ]
        ),
        tools=[],
        middleware=[
            # Without this the supervisor has no `files` channel and every
            # finding a specialist writes is dropped on the way back.
            FilesystemMiddleware(backend=build_supervisor_backend(InMemoryStore())),
            SubAgentMiddleware(
                backend=build_supervisor_backend(InMemoryStore()),
                subagents=[
                    {"name": "writer", "description": "writes", "runnable": writer},
                    {"name": "reader", "description": "reads", "runnable": reader},
                ],
                system_prompt=None,
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    result = supervisor.invoke(
        {"messages": [("user", "go")]},
        config={"configurable": {"thread_id": "cross-delegation"}},
    )

    files = result.get("files") or {}
    assert path in files, "the first delegation's finding never reached the conversation"
    assert "READ BY THE SECOND SPECIALIST" in str(files[path]["content"]), (
        "the second delegation could not open the first one's finding"
    )


# --------------------------------------------------------------------------- #
# What the supervisor is told about a job
# --------------------------------------------------------------------------- #


def _job(**overrides: Any) -> SimpleNamespace:
    created = datetime.now(UTC) - timedelta(seconds=90)
    defaults = {
        "id": uuid4(),
        "specialist": "software-specialist",
        "objective": "Survey how every indexed repository handles fitting",
        "status": "running",
        "created_at": created,
        "finished_at": None,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def test_a_finished_job_reads_as_finished_whether_it_worked_or_not() -> None:
    """One word the model acts on, not five it has to interpret.

    Every terminal status means "there is something here for the user", and the
    difference is in what `check_research` hands over. Cancellation is included
    because collecting is what clears a job off the list -- a cancelled job that
    read as its own status was never collected, so it was offered again at the
    start of every turn for the rest of the conversation.
    """
    assert _summary(_job(status="succeeded"))["status"] == "finished"
    assert _summary(_job(status="failed"))["status"] == "finished"
    assert _summary(_job(status="cancelled"))["status"] == "finished"
    assert _summary(_job(status="running"))["status"] == "running"
    assert _summary(_job(status="queued"))["status"] == "queued"


def test_a_long_objective_is_truncated_for_the_listing() -> None:
    long_objective = "x" * 200
    summary = _summary(_job(objective=long_objective))

    assert len(summary["objective"]) < 100
    assert summary["objective"].endswith("…")


def test_the_age_of_a_finished_job_stops_at_its_finish() -> None:
    created = datetime.now(UTC) - timedelta(seconds=600)
    summary = _summary(
        _job(created_at=created, finished_at=created + timedelta(seconds=120))
    )

    assert summary["age_seconds"] == 120


# --------------------------------------------------------------------------- #
# The tools themselves
# --------------------------------------------------------------------------- #


def test_the_supervisor_gets_exactly_three_research_tools() -> None:
    tools = build_research_tools(["software-specialist"])

    assert [tool.name for tool in tools] == [
        "start_research",
        "check_research",
        "list_research",
    ]


@pytest.mark.asyncio
async def test_start_refuses_a_specialist_that_does_not_exist() -> None:
    tools = {tool.name: tool for tool in build_research_tools(["software-specialist"])}
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            user_id=str(uuid4()), thread_id="t1", provider="blablador", model="m"
        ),
        tool_call_id="call-1",
    )

    answer = await tools["start_research"].coroutine(
        specialist="nonexistent-specialist",
        objective="anything",
        runtime=runtime,
    )

    assert "no specialist called" in answer


def test_a_run_without_an_authenticated_user_starts_nothing() -> None:
    """The owner comes from runtime context, never from a tool argument.

    A user id the model could supply would be an ownership check the model
    controls, which is not a check.
    """
    from juena.tools.research import _identity

    assert _identity(SimpleNamespace(context=SimpleNamespace())) is None
    assert (
        _identity(SimpleNamespace(context=SimpleNamespace(user_id="not-a-uuid", thread_id="t")))
        is None
    )
    user_id = uuid4()
    assert _identity(
        SimpleNamespace(context=SimpleNamespace(user_id=str(user_id), thread_id="t"))
    ) == (user_id, "t")


# --------------------------------------------------------------------------- #
# Collecting a terminal job, whatever its outcome
# --------------------------------------------------------------------------- #


def _collector(monkeypatch: Any, job: SimpleNamespace) -> list:
    """`check_research` wired to one job, recording every `mark_collected`."""
    from juena.tools import research as research_tools

    marked: list = []

    async def fake_get(job_id, user_id):  # noqa: ANN001, ANN202, ARG001
        return job

    async def fake_mark_collected(job_id):  # noqa: ANN001, ANN202
        marked.append(job_id)

    monkeypatch.setattr(research_tools.jobs, "get", fake_get)
    monkeypatch.setattr(research_tools.jobs, "mark_collected", fake_mark_collected)
    return marked


def _check_runtime(state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(user_id=str(uuid4()), thread_id="t1"),
        state=state or {},
        tool_call_id="call-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_an_outcome_without_findings_is_still_acknowledged_durably(
    monkeypatch: Any, status: str
) -> None:
    """A failure the user never saw is as lost as a report they never saw.

    `failed` used to call `mark_collected` *before* its message was checkpointed,
    so a crash in between left the row saying delivered and the conversation
    never seeing why. `cancelled` returned plain text and stamped nothing at all,
    so it stayed outstanding for the rest of the conversation. Both now take the
    same route as a report: the marker rides the same write as the message, and
    the row is stamped on the next call.
    """
    job = _job(status=status, error="the model provider refused", finished_at=datetime.now(UTC))
    marked = _collector(monkeypatch, job)
    tools = {tool.name: tool for tool in build_research_tools(["software-specialist"])}

    result = await tools["check_research"].coroutine(
        job_id=str(job.id), runtime=_check_runtime()
    )

    assert result.update["delivered_research_job_ids"] == [str(job.id)]
    assert result.update["messages"][0].tool_call_id == "call-1"
    assert marked == [], "the row must not be stamped before the write is checkpointed"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
async def test_a_terminal_job_already_delivered_is_never_handed_over_twice(
    monkeypatch: Any, status: str
) -> None:
    """The marker is checked before the outcome is read, for every status.

    Reporting the same failure twice is exactly as wrong as reporting the same
    result twice, and this is where the row finally learns it was collected.
    """
    job = _job(
        status=status,
        error="boom",
        report="a report",
        findings={},
        base_findings={},
        finished_at=datetime.now(UTC),
    )
    marked = _collector(monkeypatch, job)
    tools = {tool.name: tool for tool in build_research_tools(["software-specialist"])}
    state = {"delivered_research_job_ids": [str(job.id)]}

    result = await tools["check_research"].coroutine(
        job_id=str(job.id), runtime=_check_runtime(state)
    )

    assert isinstance(result, str)
    assert "already reported" in result
    assert marked == [job.id]


# --------------------------------------------------------------------------- #
# A run that returns is not the same as a run that answered
# --------------------------------------------------------------------------- #


def _runner_harness(monkeypatch: Any, result: dict) -> list[dict]:
    """Wire `runner._run` to a specialist returning `result`, recording outcomes."""
    from juena.server.agent import registry

    outcomes: list[dict] = []

    class _Specialist:
        async def ainvoke(self, _payload: Any, **_kwargs: Any) -> dict:
            return result

    async def fake_get_agent(_name: str) -> Any:
        async def aget_state(_config: Any) -> Any:
            return SimpleNamespace(values={})

        return SimpleNamespace(aget_state=aget_state)

    async def fake_mark_running(_job_id: Any) -> None:
        return None

    async def fake_finish(job_id: Any, **fields: Any) -> None:
        outcomes.append({"job_id": job_id, **fields})

    resources = SimpleNamespace(background_specialists={"software-specialist": _Specialist()})
    monkeypatch.setattr(registry, "get_agent", fake_get_agent)
    monkeypatch.setattr(registry, "get_agent_resources", lambda _name: resources)
    monkeypatch.setattr(runner.jobs, "mark_running", fake_mark_running)
    monkeypatch.setattr(runner.jobs, "finish", fake_finish)
    return outcomes


async def _run_once(monkeypatch: Any, result: dict) -> dict:
    outcomes = _runner_harness(monkeypatch, result)
    await runner._run(
        uuid4(),
        "survey the fitting code",
        "t1",
        uuid4(),
        "software-specialist",
        "blablador",
        "a-model",
        timeout_seconds=30,
    )
    assert len(outcomes) == 1
    return outcomes[0]


def _handoff_package(structured_response: Any) -> AIMessage:
    """The real hand-off message, composed by the real middleware.

    Not a hand-written fixture. The first version of these tests asserted on a
    result carrying only a `HumanMessage`, a shape production never produces, and
    passed while the defect it was written for was untouched:
    `SpecialistOutcomeMiddleware` *always* emits a non-empty `AIMessage`, so the
    "no final message" branch it exercised is unreachable in a real run.

    Composing the package through the middleware means the test cannot drift from
    what the runner will actually be handed.
    """
    update = SpecialistOutcomeMiddleware(specialist_name="software-specialist").after_agent(
        {"structured_response": structured_response},
        SimpleNamespace(),
    )
    return update["messages"][0]


def test_the_middleware_states_its_verdict_as_a_boolean() -> None:
    """The premise both runner tests below rest on.

    Asserted directly so that a change in the middleware fails here, naming the
    cause, rather than only in the runner tests, which would look like a runner
    bug.
    """
    unverified = _handoff_package(None)
    verified = _handoff_package(
        {
            "status": "completed",
            "finding": "silx fits with leastsq",
            "evidence": ["leastsq.py:40"],
        }
    )

    assert "None returned." in unverified.text
    assert "STATUS: UNVERIFIED" in unverified.text
    assert len(unverified.text) > 400, "the failing package is long prose, not an empty string"
    assert outcome_verified(unverified) is False

    assert "STATUS: VERIFIED" in verified.text
    assert outcome_verified(verified) is True

    assert outcome_verified(AIMessage(content="never went through the middleware")) is False


@pytest.mark.asyncio
async def test_an_unverified_package_is_recorded_as_failed(monkeypatch: Any) -> None:
    """`ainvoke` returning says the graph ended, not that the specialist answered.

    This is the observed failure, not a hypothetical. With the RAG store answering
    500 to every search, three runs ended normally having found nothing, and each
    stored `succeeded` — because the check was "is the final message non-empty",
    and the middleware's `None returned. / STATUS: UNVERIFIED` package is 543
    bytes of perfectly non-empty prose saying there is no result.

    So this passes the *real* package through, and the assertion is on the stored
    status. A test built on a message the middleware cannot emit proves nothing.
    """
    outcome = await _run_once(
        monkeypatch,
        {
            "messages": [
                HumanMessage(content="survey the fitting code"),
                _handoff_package(None),
            ],
            "files": {"/findings/half-written.md": {"content": "notes, no conclusion"}},
        },
    )

    assert outcome["status"] == "failed"
    assert outcome["error"] == runner.NO_REPORT
    assert "findings" not in outcome, (
        "a run that could not state a conclusion must not write into the "
        "conversation's findings"
    )
    assert "STATUS: UNVERIFIED" in outcome["report"], (
        "the server's account of why it failed is the thing that made this class "
        "of failure diagnosable; it is kept on the row"
    )


@pytest.mark.asyncio
async def test_a_verified_package_is_recorded_as_succeeded_with_its_findings(
    monkeypatch: Any,
) -> None:
    """The other half of the branch, so the fix cannot quietly fail every run."""
    outcome = await _run_once(
        monkeypatch,
        {
            "messages": [
                HumanMessage(content="survey the fitting code"),
                _handoff_package(
                    {
                        "status": "completed",
                        "finding": "silx fits with leastsq",
                        "evidence": ["leastsq.py:40"],
                    }
                ),
            ],
            "files": {"/findings/fitting.md": {"content": "paths and line numbers"}},
        },
    )

    assert outcome["status"] == "succeeded"
    assert "silx fits with leastsq" in outcome["report"]
    assert set(outcome["findings"]) == {"/findings/fitting.md"}


# --------------------------------------------------------------------------- #
# Acknowledgement does not wait on the model
# --------------------------------------------------------------------------- #


def test_the_delivery_middleware_registers_an_after_agent_hook() -> None:
    """The same check `create_agent` makes when it decides to build the node.

    Renaming or dropping `aafter_agent` would not fail anything else: the
    middleware would still compile, still declare the marker, and quietly stop
    reconciling -- which is the state this fix was written to leave.
    """
    from langchain.agents.middleware import AgentMiddleware

    from juena.research.delivery import ResearchDeliveryMiddleware

    assert ResearchDeliveryMiddleware.aafter_agent is not AgentMiddleware.aafter_agent


@pytest.mark.asyncio
async def test_a_delivered_job_is_acknowledged_without_the_model_asking(
    monkeypatch: Any,
) -> None:
    """Reconciliation used to need `list_research`, and the model skips it.

    Observed directly: a job was delivered, read aloud, and edited by a later
    specialist, and the next turn called no research tool at all. The row kept
    `collected_at` null, so `GET /research` went on reporting `ready: 1` for a
    result the user already had.
    """
    from juena.research import delivery

    stamped: list = []

    async def fake_mark_collected_many(job_ids: Any) -> int:
        stamped.extend(job_ids)
        return len(stamped)

    monkeypatch.setattr(delivery.jobs, "mark_collected_many", fake_mark_collected_many)

    job_id = uuid4()
    await delivery.ResearchDeliveryMiddleware().aafter_agent(
        {"delivered_research_job_ids": [str(job_id)], "messages": []},
        SimpleNamespace(),
    )

    assert stamped == [job_id]


def test_a_malformed_marker_is_skipped_rather_than_ending_the_turn() -> None:
    from juena.research.delivery import delivered_uuids

    job_id = uuid4()

    assert delivered_uuids({"delivered_research_job_ids": [str(job_id), "not-an-id"]}) == [job_id]
    assert delivered_uuids({}) == []


@pytest.mark.asyncio
async def test_a_turn_that_delivered_nothing_opens_no_transaction() -> None:
    """The hook runs on every turn, so the common case must cost nothing.

    There is no database configured here: reaching the session factory at all
    would raise rather than return 0.
    """
    from juena.research import jobs

    assert await jobs.mark_collected_many([]) == 0
