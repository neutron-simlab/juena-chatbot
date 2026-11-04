"""Typed TIFF tools preserve sandbox evidence and cannot accept shell commands."""

import json
import shlex
from types import SimpleNamespace

from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
import pytest

from juena.tools import detector
from juena_core.agents.specialist_outcome import SpecialistOutcomeState
from juena_core.sandbox.evidence import record_sandbox_execution
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext


def graph():
    builder = StateGraph(SpecialistOutcomeState, context_schema=RuntimeModelContext)
    builder.add_node("tools", ToolNode([detector.analyze_tiff]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


@pytest.mark.asyncio
async def test_agent_tool_executes_fixed_request_and_updates_trusted_execution_state(monkeypatch):
    commands = []
    class Backend:
        async def aexecute(self, command):
            commands.append(command)
            record_sandbox_execution(command=command, status="completed", exit_code=0)
            return ExecuteResponse(output='{"mean": 12.5}', exit_code=0)
    monkeypatch.setattr(detector, "sandbox_enabled", lambda: True)
    monkeypatch.setattr(detector, "RuntimePodmanSandboxBackend", Backend)
    path = "/workspace/uploads/sample's $(touch should-not-run).tif"
    result = await graph().ainvoke({"messages": [AIMessage(content="", tool_calls=[{
        "name": "analyze_tiff", "id": "tiff-call", "args": {"operation": "inspect", "paths": [path]},
    }])]}, context=RuntimeModelContext(provider="blablador", model="test", run_id="run-1", user_id="alice", thread_id="thread-1"))
    assert len(commands) == 1
    argv = shlex.split(commands[0])
    assert argv[:2] == ["python", "-c"]
    assert json.loads(argv[3])["inputs"] == [path]
    assert result["messages"][-1].content == '{"mean": 12.5}'
    assert result["messages"][-1].status == "success"
    assert result["messages"][-1].tool_call_id == "tiff-call"
    assert result["execution_events"][0]["graph_run_id"] == "run-1"
    assert result["execution_events"][0]["exit_code"] == 0


@pytest.mark.asyncio
async def test_unavailable_execution_retains_input_and_never_constructs_backend(monkeypatch):
    monkeypatch.setattr(detector, "sandbox_enabled", lambda: False)
    monkeypatch.setattr(detector, "RuntimePodmanSandboxBackend", lambda: pytest.fail("No backend expected"))
    result = await detector.analyze_tiff.coroutine(
        "inspect", ["/workspace/uploads/sample.tif"], SimpleNamespace(tool_call_id="call"),
    )
    assert result.status == "error" and "original is retained" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/etc/passwd.tif", "/workspace/../private.tif", "/workspace/a.dat"])
async def test_tool_refuses_paths_outside_conversation_before_submission(monkeypatch, path):
    monkeypatch.setattr(detector, "sandbox_enabled", lambda: True)
    monkeypatch.setattr(detector, "RuntimePodmanSandboxBackend", lambda: pytest.fail("No backend expected"))
    with pytest.raises(ValueError):
        await detector.analyze_tiff.coroutine("inspect", [path], SimpleNamespace(tool_call_id="call"))


@pytest.mark.asyncio
async def test_successful_text_without_backend_evidence_is_not_verified(monkeypatch):
    class Backend:
        async def aexecute(self, command):
            return ExecuteResponse(output="looks successful", exit_code=0)
    monkeypatch.setattr(detector, "sandbox_enabled", lambda: True)
    monkeypatch.setattr(detector, "RuntimePodmanSandboxBackend", Backend)
    result = await detector.analyze_tiff.coroutine(
        "inspect", ["/workspace/uploads/sample.tif"], SimpleNamespace(tool_call_id="call", context={"run_id": "run-1"}),
    )
    assert result.update["messages"][0].status == "error"
    assert result.update["execution_events"][0]["status"] == "tool_error"


def test_tool_schema_hides_runtime_and_exposes_operations_without_code():
    fields = detector.analyze_tiff.tool_call_schema.model_json_schema()["properties"]
    assert "runtime" not in fields and "command" not in fields
    assert fields["operation"]["enum"] == ["inspect", "histogram", "roi", "compare"]
    assert fields["paths"]["maxItems"] == 5
