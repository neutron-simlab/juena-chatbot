"""Cutting the repeated-tool-call loop that strands a run without a report."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from juena.agents.loop_guard import RepeatedToolCallMiddleware


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id}


def _request(call: dict, messages: list) -> SimpleNamespace:
    return SimpleNamespace(tool_call=call, state={"messages": messages})


def _history(previous: dict, result: str = "ok") -> list:
    return [
        HumanMessage("Plot a sine wave."),
        AIMessage(content="", tool_calls=[previous]),
        ToolMessage(result, tool_call_id=previous["id"]),
    ]


def _handler(seen: list):
    def run(request):
        seen.append(request.tool_call["id"])
        return ToolMessage("ran", tool_call_id=request.tool_call["id"])

    return run


def test_an_identical_repeat_is_refused_without_running_the_tool() -> None:
    middleware = RepeatedToolCallMiddleware()
    previous = _call("write_todos", {"todos": [{"content": "Plot", "status": "completed"}]}, "a")
    repeat = _call("write_todos", {"todos": [{"content": "Plot", "status": "completed"}]}, "b")
    seen: list[str] = []

    result = middleware.wrap_tool_call(_request(repeat, _history(previous)), _handler(seen))

    assert seen == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "already made" in result.text


def test_argument_order_does_not_disguise_a_repeat() -> None:
    middleware = RepeatedToolCallMiddleware()
    previous = _call("search", {"query": "sans", "repo_id": "all"}, "a")
    repeat = _call("search", {"repo_id": "all", "query": "sans"}, "b")
    seen: list[str] = []

    result = middleware.wrap_tool_call(_request(repeat, _history(previous)), _handler(seen))

    assert seen == []
    assert result.status == "error"


def test_a_different_call_runs_normally() -> None:
    middleware = RepeatedToolCallMiddleware()
    previous = _call("search", {"query": "sans"}, "a")
    different = _call("search", {"query": "sesans"}, "b")
    seen: list[str] = []

    result = middleware.wrap_tool_call(_request(different, _history(previous)), _handler(seen))

    assert seen == ["b"]
    assert result.status != "error"


def test_repeating_a_call_from_two_turns_back_is_allowed() -> None:
    """Only the immediately preceding call is a loop; revisiting a tool is not."""

    middleware = RepeatedToolCallMiddleware()
    first = _call("ls", {"path": "/workspace"}, "a")
    between = _call("read_file", {"file_path": "/workspace/run.py"}, "b")
    again = _call("ls", {"path": "/workspace"}, "c")
    messages = [
        *_history(first),
        AIMessage(content="", tool_calls=[between]),
        ToolMessage("file body", tool_call_id=between["id"]),
    ]
    seen: list[str] = []

    result = middleware.wrap_tool_call(_request(again, messages), _handler(seen))

    assert seen == ["c"]
    assert result.status != "error"


def test_the_first_call_of_a_run_is_never_blocked() -> None:
    middleware = RepeatedToolCallMiddleware()
    seen: list[str] = []

    result = middleware.wrap_tool_call(
        _request(_call("ls", {"path": "/workspace"}, "a"), [HumanMessage("Start")]),
        _handler(seen),
    )

    assert seen == ["a"]
    assert result.status != "error"
