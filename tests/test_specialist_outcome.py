"""The research package a specialist hands back, and what the server verifies."""

import io
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from juena_core.schema.agents import SpecialistReport
from juena_core.agents.specialist_outcome import SpecialistOutcomeMiddleware
from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    value = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(value)
    yield value
    set_artifact_store_for_tests(None)


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(user_id="user-a", thread_id="thread-a"),
        config={"run_id": "run-1"},
    )


def _state(**updates) -> dict:
    state = {
        "messages": [HumanMessage("Explain the result.")],
        "structured_response": SpecialistReport(
            status="completed",
            finding="Verified finding.",
            evidence=["Direct test evidence"],
        ),
    }
    state.update(updates)
    return state


def _package(middleware, state) -> str:
    update = middleware.after_agent(state, _runtime())
    assert update["structured_response"] is None
    return update["messages"][0].text


def test_progress_sentence_without_structured_report_fails_closed(store) -> None:
    """The original incident: a run that died mid-sentence must not read as a result."""

    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")
    state = {
        "messages": [
            HumanMessage("Use JScatter to fit data and generate sas_fit.png."),
            AIMessage("Now let me check the jscatter skill documentation:"),
        ]
    }

    package = _package(middleware, state)

    assert "STATUS: UNVERIFIED" in package
    assert "returned no structured report" in package
    assert "Result artifacts delivered: none." in package
    assert "Do not report results, values, plots, or file names" in package
    # The stranded sentence must not be forwarded as if it were the answer.
    assert "Now let me check" not in package


def test_report_and_server_facts_travel_together(store) -> None:
    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")

    package = _package(middleware, _state())

    assert "<specialist_report>" in package and "</specialist_report>" in package
    assert "<verified_by_server>" in package and "</verified_by_server>" in package
    assert "Finding: Verified finding." in package
    assert "STATUS: VERIFIED" in package
    assert package.index("<specialist_report>") < package.index("<verified_by_server>")


def test_a_run_where_every_command_failed_is_unverified(store) -> None:
    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")
    state = _state(
        sandbox_execution_events=[
            {"command": "python run.py", "status": "failed", "exit_code": 1}
        ],
    )

    failed = _package(middleware, state)
    state["sandbox_execution_events"].append(
        {"command": "python run.py", "status": "completed", "exit_code": 0}
    )
    recovered = _package(middleware, state)

    assert "STATUS: UNVERIFIED" in failed
    assert "Every sandbox command in this run failed" in failed
    # A failed first attempt followed by a successful retry is a success.
    assert "STATUS: VERIFIED" in recovered
    assert "failed (exit 1), completed (exit 0)" in recovered


def test_a_refused_export_alone_does_not_condemn_the_run(store) -> None:
    """`skipped` never reached the worker, so it is not a failed command."""

    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")
    state = _state(
        sandbox_execution_events=[
            {"command": "base64 /workspace/outputs/p.png", "status": "skipped", "exit_code": 0}
        ],
    )

    package = _package(middleware, state)

    assert "STATUS: VERIFIED" in package
    assert "Every sandbox command in this run failed" not in package


def test_only_artifacts_from_this_run_are_reported(store) -> None:
    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="old",
        filename="earlier.png",
        content=_png(),
    )
    baseline = middleware.before_agent({}, _runtime())
    state = _state(**baseline)

    before = _package(middleware, state)
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="sas_fit.png",
        content=_png(),
    )
    after = _package(middleware, state)

    assert "Result artifacts delivered: none." in before
    assert "sas_fit.png (image" in after
    assert "earlier.png" not in after


def test_undelivered_output_is_reported_and_left_for_the_root(store) -> None:
    middleware = SpecialistOutcomeMiddleware(specialist_name="software-specialist")
    store.note_undelivered("user-a", "thread-a", "fit.png", "invalid PNG")

    package = _package(middleware, _state())

    assert "Produced but NOT delivered" in package
    assert "fit.png: invalid PNG" in package
    assert "fit.png was not delivered: invalid PNG" in package
    # Peeked, never drained: the root response boundary still owns delivery.
    assert store.drain_undelivered("user-a", "thread-a") == [("fit.png", "invalid PNG")]


def test_blocked_status_is_passed_through_rather_than_failed(store) -> None:
    middleware = SpecialistOutcomeMiddleware(specialist_name="neutron-research-specialist")
    state = _state(
        structured_response=SpecialistReport(
            status="blocked",
            finding="The staged file has no q column.",
        )
    )

    package = _package(middleware, state)

    assert "STATUS: VERIFIED" in package
    assert "Status: blocked" in package


def test_real_agent_delivers_the_package_as_its_final_message(store) -> None:
    """Clearing `structured_response` must route the package through the message path."""

    class _ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    agent = create_agent(
        model=_ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "SpecialistReport",
                            "args": {
                                "status": "completed",
                                "finding": "Verified finding.",
                                "evidence": ["Direct evidence"],
                                "actions": [],
                                "limitations": [],
                            },
                            "id": "structured-report",
                        }
                    ],
                )
            ]
        ),
        tools=[],
        response_format=ToolStrategy(SpecialistReport),
        middleware=[SpecialistOutcomeMiddleware(specialist_name="specialist")],
    )

    result = agent.invoke({"messages": [("user", "Explain the result.")]})

    assert result["structured_response"] is None
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "Finding: Verified finding." in final.text
    assert "STATUS: VERIFIED" in final.text
