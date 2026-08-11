"""Approval records written around sandbox tool execution."""

import io
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from PIL import Image

from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.middleware import SandboxExecutionMiddleware
from juena.sandbox.evidence import record_sandbox_execution


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")
    set_artifact_store_for_tests(store)
    yield store
    set_artifact_store_for_tests(None)


def _request(command: str, *, name: str = "execute", user_id: str = "user-a"):
    return SimpleNamespace(
        tool_call={"name": name, "args": {"command": command}, "id": "call-1"},
        runtime=SimpleNamespace(
            context=SimpleNamespace(user_id=user_id, thread_id="thread-a"),
            config={"run_id": "run-1"},
        ),
    )


def _handler(
    text: str = "plot written\n",
    *,
    status: str = "completed",
    exit_code: int | None = 0,
):
    def handle(request):
        record_sandbox_execution(
            command=request.tool_call["args"]["command"],
            status=status,
            exit_code=exit_code,
        )
        return ToolMessage(content=text, tool_call_id="call-1")

    return handle


def _message(result) -> ToolMessage:
    if isinstance(result, ToolMessage):
        return result
    assert isinstance(result, Command)
    return next(
        item for item in result.update["messages"] if isinstance(item, ToolMessage)
    )


def _filenames(store) -> list[str]:
    return [item["filename"] for item in store.claim_for_message("user-a", "thread-a")]


def test_approved_command_and_output_are_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"
    assert _message(result).status == "success"
    assert result.update["sandbox_execution_events"][0]["exit_code"] == 0
    assert _filenames(store) == ["generated-command.sh", "execution-output.txt"]


def test_many_approved_commands_leave_room_for_results(store) -> None:
    """Records are the application's bookkeeping and must not spend the user's budget.

    Six approved commands used to write twelve records into a twelve-file
    allowance, so the figure the turn was about had nowhere to go.
    """
    middleware = SandboxExecutionMiddleware()
    for index in range(6):
        middleware.wrap_tool_call(_request(f"python step_{index}.py"), _handler())

    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="fit.png",
        content=_png(),
    )

    assert plot.kind == "image"


def test_undelivered_write_waits_for_the_specialist_outcome(store) -> None:
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"
    assert store.peek_undelivered("user-a", "thread-a") == [
        ("report.html", "type is not allowed")
    ]
    second = middleware.wrap_tool_call(_request("python again.py"), _handler())
    assert _message(second).text == "plot written\n"
    assert store.peek_undelivered("user-a", "thread-a") == [
        ("report.html", "type is not allowed")
    ]


def test_execution_record_matches_what_the_model_was_shown(store) -> None:
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python plot.py"), _handler())

    saved = store.claim_for_message("user-a", "thread-a")
    output_ref = next(item for item in saved if item["filename"] == "execution-output.txt")
    _ref, content = store.get("user-a", output_ref["artifact_id"])
    assert content == b"plot written\n"


def test_a_command_with_nothing_queued_is_left_alone(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"


def test_edited_command_is_what_gets_recorded(store) -> None:
    """The case the old exact-text lookup could not handle.

    The human edited the command before approving, so `wrap_tool_call` receives
    the edited text. Nothing is matched against the originally proposed command.
    """
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python revised.py --seed 7"), _handler())

    saved = store.claim_for_message("user-a", "thread-a")
    command_ref = next(item for item in saved if item["filename"] == "generated-command.sh")
    _ref, content = store.get("user-a", command_ref["artifact_id"])
    assert content == b"python revised.py --seed 7\n"


@pytest.mark.asyncio
async def test_async_path_records_the_same_artifacts(store) -> None:
    middleware = SandboxExecutionMiddleware()

    async def handle(request):
        record_sandbox_execution(
            command=request.tool_call["args"]["command"],
            status="completed",
            exit_code=0,
        )
        return ToolMessage(content="done\n", tool_call_id="call-1")

    await middleware.awrap_tool_call(_request("python plot.py"), handle)

    assert _filenames(store) == ["generated-command.sh", "execution-output.txt"]


def test_other_tools_are_not_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python plot.py", name="write_file"), _handler())

    assert _filenames(store) == []


def test_redundant_artifact_export_is_not_recorded(store) -> None:
    """Commands that never reach the approval gate get no approval record."""
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(
        _request("base64 /workspace/outputs/plot.png"), _handler()
    )

    assert _filenames(store) == []


def test_unauthenticated_run_is_not_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()
    request = _request("python plot.py")
    request.runtime.context = SimpleNamespace(user_id=None, thread_id=None)

    result = middleware.wrap_tool_call(request, _handler())

    assert _message(result).text == "plot written\n"
    assert _filenames(store) == []


def test_nonzero_execution_is_error_and_preserves_typed_evidence(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(
        _request("python broken.py"),
        _handler("traceback", status="failed", exit_code=1),
    )

    assert isinstance(result, Command)
    assert _message(result).status == "error"
    assert result.update["sandbox_execution_events"] == [
        {
            "command": "python broken.py",
            "status": "failed",
            "exit_code": 1,
            "truncated": False,
            "artifact_ids": [],
            "artifact_filenames": [],
            "dropped": [],
        }
    ]


def test_missing_backend_evidence_is_a_tool_error(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(
        _request("python unknown.py"),
        lambda _request: ToolMessage(content="unknown", tool_call_id="call-1"),
    )

    assert _message(result).status == "error"
    assert result.update["sandbox_execution_events"][0]["status"] == "tool_error"


def test_tool_still_runs_when_the_record_cannot_be_saved(store) -> None:
    """An unwritable artifact store must not block the command."""
    middleware = SandboxExecutionMiddleware()
    store.root.write_text("not a directory", encoding="utf-8")

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert _message(result).text == "plot written\n"
