"""Cutting the repeated-tool-call loop that strands a run without a report."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from juena_core.agents.loop_guard import GUARD_FLAG, REPEAT_ERROR, RepeatedToolCallMiddleware


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


@pytest.mark.asyncio
async def test_the_async_path_refuses_a_repeat_too() -> None:
    """The server runs `ainvoke`, so the sync-only tests proved nothing about it."""

    middleware = RepeatedToolCallMiddleware()
    previous = _call("read_file", {"file_path": "/a.dat"}, "a")
    repeat = _call("read_file", {"file_path": "/a.dat"}, "b")
    seen: list[str] = []

    async def handler(request):  # noqa: ANN001, ANN202
        seen.append(request.tool_call["id"])
        return ToolMessage("ran", tool_call_id=request.tool_call["id"])

    result = await middleware.awrap_tool_call(_request(repeat, _history(previous)), handler)

    assert seen == []
    assert result.status == "error"


# --------------------------------------------------------------------------- #
# Ending the run, not just refusing the call
# --------------------------------------------------------------------------- #


def _refusal(name: str, call_id: str) -> ToolMessage:
    return ToolMessage(
        REPEAT_ERROR.format(name=name),
        name=name,
        tool_call_id=call_id,
        status="error",
        additional_kwargs={GUARD_FLAG: True},
    )


def test_one_refusal_is_a_nudge_and_does_not_end_the_run() -> None:
    """A model that can act on the message deserves the chance to."""

    middleware = RepeatedToolCallMiddleware()
    call = _call("read_file", {"file_path": "/a.dat"}, "b")
    messages = [
        *_history(_call("read_file", {"file_path": "/a.dat"}, "a")),
        AIMessage(content="", tool_calls=[call]),
        _refusal("read_file", "b"),
    ]

    assert middleware.before_model({"messages": messages}, None) is None


def test_a_second_refusal_ends_the_run() -> None:
    """Two ignored refusals are evidence the message is not landing.

    The live failure this replaces: one `read_file` re-issued 60 times after it
    had already succeeded, each refusal costing a model call and changing
    nothing. Three identical calls is the new ceiling.
    """
    middleware = RepeatedToolCallMiddleware()
    messages = [*_history(_call("read_file", {"file_path": "/a.dat"}, "a"))]
    for call_id in ("b", "c"):
        messages.append(
            AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "/a.dat"}, call_id)])
        )
        messages.append(_refusal("read_file", call_id))

    update = middleware.before_model({"messages": messages}, None)

    assert update is not None
    assert update["jump_to"] == "end"
    assert "Stopped after 3 identical calls to `read_file`" in update["messages"][0].text


def test_refusals_from_earlier_in_a_healthy_run_do_not_accumulate() -> None:
    """Only an unbroken tail counts, or a run that recovered twice would be killed."""

    middleware = RepeatedToolCallMiddleware()
    messages = [
        *_history(_call("ls", {"path": "/workspace"}, "a")),
        AIMessage(content="", tool_calls=[_call("ls", {"path": "/workspace"}, "b")]),
        _refusal("ls", "b"),
        # the model took the hint, did something else, and got a real result
        AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "/run.py"}, "c")]),
        ToolMessage("body", tool_call_id="c"),
        AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "/run.py"}, "d")]),
        _refusal("read_file", "d"),
    ]

    assert middleware.before_model({"messages": messages}, None) is None


@pytest.mark.asyncio
async def test_a_locked_model_is_stopped_rather_than_argued_with() -> None:
    """End to end, through a real graph, because the parts were never the problem.

    Every unit above passed while the live system burned 59 model calls on one
    repeated read. What was missing was a test that let a model keep going.
    """
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool as make_tool

    ran: list[str] = []

    @make_tool
    def read_file(file_path: str) -> str:
        """Read a file."""
        ran.append(file_path)
        return "line 1\nline 2"

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    agent = create_agent(
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "read_file",
                        "args": {"file_path": "/a.dat"},
                        "id": f"call-{index}",
                        "type": "tool_call",
                    }],
                )
                for index in range(12)
            ]
        ),
        tools=[read_file],
        middleware=[RepeatedToolCallMiddleware()],
    )
    result = await agent.ainvoke({"messages": [("user", "read it")]})

    assert ran == ["/a.dat"], "the tool ran more than once"
    assert "Stopped after 3 identical calls" in str(result["messages"][-1].content)
