"""Approval records written around sandbox tool execution."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from juena.sandbox.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena.sandbox.middleware import SandboxExecutionMiddleware


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
