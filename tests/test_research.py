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

    assert "Status: OK" in runner._report_text(result)


def test_a_run_that_reported_nothing_says_so_rather_than_returning_empty() -> None:
    assert runner._report_text({"messages": []}) == runner.NO_REPORT


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
        }
    )

    assert set(written) == {"/findings/sasview-fitting.md"}


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


def test_findings_never_go_behind_a_composite_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/findings/` must stay on the default backend, never become a route.

    Routing it to a `StateBackend` looks like the obvious way to make it mean one
    place regardless of the sandbox. It is not: `CompositeBackend` strips the
    matched prefix, so `/findings/survey.md` would be stored under `/survey.md`,
    where nothing can tell a finding from any other top-level file -- and the
    runner's own filtering would stop matching. This pins the behaviour that
    makes the shortcut wrong, so it is not rediscovered by trying it.
    """
    from deepagents.backends import StateBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from juena.agents.backends import FINDINGS_PREFIX
    from juena.agents.specialist_runtime import build_specialist_backend

    monkeypatch.setattr(sandbox_runtime.global_config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(sandbox_runtime, "RuntimePodmanSandboxBackend", lambda: object())
    sandboxed = sandbox_runtime.build_sandbox_backend(
        repo_cache_root=None, skills_dir=None, has_skills=False
    )
    assert FINDINGS_PREFIX not in build_specialist_backend().routes
    assert FINDINGS_PREFIX not in sandboxed.routes

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    routed = create_agent(
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_file",
                        "args": {
                            "file_path": "/findings/survey.md",
                            "content": "silx uses leastsq",
                        },
                        "id": "write",
                        "type": "tool_call",
                    }],
                ),
                AIMessage(content="done"),
            ]
        ),
        tools=[],
        middleware=[
            FilesystemMiddleware(
                backend=CompositeBackend(
                    default=StateBackend(), routes={FINDINGS_PREFIX: StateBackend()}
                )
            )
        ],
        checkpointer=InMemorySaver(),
    )
    result = routed.invoke(
        {"messages": [("user", "record a finding")]},
        config={"configurable": {"thread_id": "routed-findings"}},
    )

    assert list(result.get("files") or {}) == ["/survey.md"], (
        "a route strips its prefix, which is exactly why /findings must not be one"
    )


def test_a_background_run_without_a_sandbox_keeps_the_findings_prefix() -> None:
    """The background specialist's `/findings/` is graph state, keys intact.

    It has no sandbox, so `/findings/` falls to the default state backend and the
    full path survives -- which is what `_written_findings` filters on.
    """
    from juena.agents.backends import FINDINGS_PREFIX
    from juena.agents.specialist_runtime import build_specialist_backend

    backend = build_specialist_backend()
    assert FINDINGS_PREFIX not in backend.routes
    assert runner.FINDINGS_PREFIX == FINDINGS_PREFIX


def test_findings_ride_back_inside_the_report() -> None:
    """The supervisor cannot open a `/findings/` path, so it gets the text.

    Its own backend serves `/memories/` only, and a specialist it delegates to
    afterwards may be reading `/findings/` from a sandbox workspace the
    background run never touched. The content is the one form that always works.
    """
    from juena.tools.research import _report_with_findings

    job = SimpleNamespace(
        report="<specialist_report>\nStatus: OK\n</specialist_report>",
        findings={"/findings/survey.md": {"content": "silx uses leastsq.py at line 40"}},
    )
    combined = _report_with_findings(job)

    assert "Status: OK" in combined
    assert "/findings/survey.md" in combined
    assert "silx uses leastsq.py at line 40" in combined

    bare = SimpleNamespace(report="just the report", findings={})
    assert _report_with_findings(bare) == "just the report"


def test_a_very_long_finding_is_truncated_before_it_enters_history() -> None:
    from juena.tools.research import _FINDINGS_PREVIEW, _report_with_findings

    job = SimpleNamespace(
        report="r", findings={"/findings/huge.md": {"content": "x" * (_FINDINGS_PREVIEW * 3)}}
    )
    combined = _report_with_findings(job)

    assert "truncated" in combined
    assert len(combined) < _FINDINGS_PREVIEW * 2


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

    `succeeded` and `failed` both mean "there is something here for the user",
    and the difference is in the report `check_research` hands over.
    """
    assert _summary(_job(status="succeeded"))["status"] == "finished"
    assert _summary(_job(status="failed"))["status"] == "finished"
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
