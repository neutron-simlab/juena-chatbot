"""Approval records written around sandbox tool execution."""

import io
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from PIL import Image

from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.middleware import SandboxExecutionMiddleware


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


def _handler(text: str = "plot written\n"):
    def handle(request):
        return ToolMessage(content=text, tool_call_id="call-1")

    return handle


def _filenames(store) -> list[str]:
    return [item["filename"] for item in store.claim_for_message("user-a", "thread-a")]


def test_approved_command_and_output_are_recorded(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert result.text == "plot written\n"
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


def test_undelivered_write_is_reported_on_the_next_approved_execution(store) -> None:
    """`write_file` has no warning channel, so a later execution carries it early.

    Reporting it as a failed write would be false -- the bytes are on disk --
    and would send the model into a rewrite that collides with its own file.
    """
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert "plot written" in result.text
    assert "NOT delivered to the user" in result.text
    assert "report.html: type is not allowed" in result.text
    # Drained, so a second command does not repeat it.
    second = middleware.wrap_tool_call(_request("python again.py"), _handler())
    assert "NOT delivered" not in second.text


def test_execution_record_matches_what_the_model_was_shown(store) -> None:
    store.note_undelivered("user-a", "thread-a", "report.html", "type is not allowed")
    middleware = SandboxExecutionMiddleware()

    middleware.wrap_tool_call(_request("python plot.py"), _handler())

    saved = store.claim_for_message("user-a", "thread-a")
    output_ref = next(item for item in saved if item["filename"] == "execution-output.txt")
    _ref, content = store.get("user-a", output_ref["artifact_id"])
    assert b"NOT delivered to the user" in content


def test_a_command_with_nothing_queued_is_left_alone(store) -> None:
    middleware = SandboxExecutionMiddleware()

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert result.text == "plot written\n"


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

    assert result.text == "plot written\n"
    assert _filenames(store) == []


def test_tool_still_runs_when_the_record_cannot_be_saved(store) -> None:
    """An unwritable artifact store must not block the command."""
    middleware = SandboxExecutionMiddleware()
    store.root.write_text("not a directory", encoding="utf-8")

    result = middleware.wrap_tool_call(_request("python plot.py"), _handler())

    assert result.text == "plot written\n"
