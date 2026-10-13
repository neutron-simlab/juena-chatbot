"""What actually reaches Langfuse, checked on the exported spans.

The contracts (src/juena/server/tracing.py): every graph invocation is its own
trace; every span in it -- not only the root -- carries the chat's thread id as
session and the user id; and concurrent chats never mix. These tests run a real
checkpointed LangGraph through the real `start_langfuse()` hook and core's real
`build_run_context`, streamed with the options core's chat endpoint uses, and
read the spans from an in-memory exporter instead of a server.

Verified against the locked langfuse SDK. An SDK upgrade that changes how
attributes propagate fails here first.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import var_child_runnable_config
from langfuse import LangfuseOtelSpanAttributes
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry import trace as otel_trace
from sse_starlette.sse import AppStatus

import juena_core.tracing as core_tracing
from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena_core.server.streaming.own_task import AgentStreamResponse

import juena.server.tracing as langfuse_tracing
from juena.research import runner
from juena.server.database.models import ResearchJob

SESSION = LangfuseOtelSpanAttributes.TRACE_SESSION_ID
USER = LangfuseOtelSpanAttributes.TRACE_USER_ID
OBSERVATION_TYPE = LangfuseOtelSpanAttributes.OBSERVATION_TYPE


def _graph(*, ask_first: bool = False):
    """A model node, optionally behind a node that interrupts to ask the user."""

    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="an answer")]))

    async def agent(state: MessagesState):
        return {"messages": [await model.ainvoke(state["messages"])]}

    async def ask_user(state: MessagesState):
        answer = interrupt("Which repository?")
        return {"messages": [HumanMessage(content=answer)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    if ask_first:
        builder.add_node("ask_user", ask_user)
        builder.add_edge(START, "ask_user")
        builder.add_edge("ask_user", "agent")
    else:
        builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())


async def _stream(graph, graph_input, *, thread_id: str, user_id: str) -> None:
    run = AgentInputHandler.build_run_context(thread_id=thread_id, user_id=user_id)
    # The options of juena_core.server.api.endpoints._stream_agent_payloads.
    async for _ in graph.astream(
        graph_input,
        config=run.config,
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True,
        version="v2",
    ):
        pass


async def _request(graph, graph_input, *, thread_id: str, user_id: str) -> None:
    """One chat or resume request: its own task, starting from an empty context.

    That is how the server runs each HTTP request, and it matters here: after
    a stream, the SDK leaves the run's root span attached to the context that
    ran it (observed on both finished and interrupted runs with the locked
    versions). A second invocation in that same context would be filed under
    the first one's trace. Each request's context ends with the request, so in
    the server that never happens -- and these tests must not hide it by
    running two invocations in one context either.
    """

    await asyncio.create_task(
        _stream(graph, graph_input, thread_id=thread_id, user_id=user_id),
        context=contextvars.Context(),
    )


def _hello() -> dict:
    return {"messages": [HumanMessage(content="hello")]}


def _by_trace(spans) -> dict[int, list]:
    traces: dict[int, list] = defaultdict(list)
    for span in spans:
        traces[span.context.trace_id].append(span)
    return traces


@pytest.mark.asyncio
async def test_every_span_of_a_streamed_run_carries_session_and_user(langfuse_capture) -> None:
    await _request(_graph(), _hello(), thread_id="thread-1", user_id="user-1")

    spans = langfuse_capture.spans()
    assert len(_by_trace(spans)) == 1
    # Not only the root: a node below it and the model call below that.
    assert any(span.parent is not None for span in spans)
    assert "generation" in {span.attributes.get(OBSERVATION_TYPE) for span in spans}
    for span in spans:
        assert span.attributes.get(SESSION) == "thread-1", span.name
        assert span.attributes.get(USER) == "user-1", span.name


@pytest.mark.asyncio
async def test_without_run_inline_child_spans_lose_attribution(
    langfuse_capture, monkeypatch
) -> None:
    """Proves the test above exercises the propagation `run_inline` fixes."""

    def without_run_inline(*, thread_id: str, user_id: str):
        tracing = langfuse_tracing._run_tracing(thread_id=thread_id, user_id=user_id)
        for handler in tracing.callbacks:
            handler.run_inline = False
        return tracing

    monkeypatch.setattr(core_tracing, "_hook", without_run_inline)

    await _request(_graph(), _hello(), thread_id="thread-1", user_id="user-1")

    spans = langfuse_capture.spans()
    assert any(span.attributes.get(SESSION) != "thread-1" for span in spans)


@pytest.mark.asyncio
async def test_answering_an_interrupt_is_a_new_trace_in_the_same_session(
    langfuse_capture,
) -> None:
    graph = _graph(ask_first=True)

    await _request(graph, _hello(), thread_id="thread-1", user_id="user-1")
    asked = {span.context.trace_id for span in langfuse_capture.spans()}
    await _request(graph, Command(resume="juena-core"), thread_id="thread-1", user_id="user-1")

    spans = langfuse_capture.spans()
    traces = _by_trace(spans)
    assert len(asked) == 1
    assert len(traces) == 2, "the resume is its own trace"
    resumed = next(spans for trace_id, spans in traces.items() if trace_id not in asked)
    assert "generation" in {span.attributes.get(OBSERVATION_TYPE) for span in resumed}
    for span in spans:
        assert span.attributes.get(SESSION) == "thread-1", span.name
        assert span.attributes.get(USER) == "user-1", span.name


@pytest.mark.asyncio
async def test_concurrent_chats_never_mix_their_identities(langfuse_capture) -> None:
    graph = _graph()
    chats = [(f"thread-{n}", f"user-{n}") for n in range(20)]

    await asyncio.gather(
        *(_request(graph, _hello(), thread_id=t, user_id=u) for t, u in chats)
    )

    traces = _by_trace(langfuse_capture.spans())
    assert len(traces) == len(chats)
    seen = set()
    for spans in traces.values():
        identities = {(span.attributes.get(SESSION), span.attributes.get(USER)) for span in spans}
        assert len(identities) == 1, identities
        seen |= identities
    assert seen == set(chats)


@pytest.mark.asyncio
async def test_lifespan_shuts_the_client_down_on_exit(langfuse_capture, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(langfuse_capture.client, "shutdown", lambda: calls.append("shutdown"))

    async with langfuse_tracing.langfuse_lifespan():
        assert calls == []

    assert calls == ["shutdown"]


# --------------------------------------------------------------------------- #
# Background research: its own trace, in the session of the chat that asked
# --------------------------------------------------------------------------- #


class _ResearchState(MessagesState):
    files: dict


def _research_graph(seen_configs: list[RunnableConfig]):
    model = GenericFakeChatModel(messages=itertools.cycle([AIMessage(content="a report")]))

    async def work(state: _ResearchState, config: RunnableConfig):
        seen_configs.append(config)
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(_ResearchState, context_schema=RuntimeModelContext)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile(checkpointer=InMemorySaver(), name="software-specialist")


def _research_job() -> ResearchJob:
    """A claimed job row, as `runner._run` hands it to `_invoke`."""
    return ResearchJob(
        id=uuid4(),
        user_id=uuid4(),
        thread_id="thread-1",
        specialist="software-specialist",
        objective="survey the fitting code",
        run_input={"files": {}, "provider": "blablador", "model": "a-model"},
        status="running",
        started_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_a_research_job_is_traced_in_the_session_of_its_chat(
    langfuse_capture,
) -> None:
    seen_configs: list[RunnableConfig] = []
    job = _research_job()

    # The poller starts every job in an empty context; so does this.
    await asyncio.create_task(
        runner._invoke(_research_graph(seen_configs), job),
        context=contextvars.Context(),
    )

    spans = langfuse_capture.spans()
    assert len(_by_trace(spans)) == 1
    # Its own graph thread, so it never writes into the chat's checkpoint...
    assert seen_configs[0]["configurable"]["thread_id"] == f"research:{job.id}"
    # ...but filed under the chat's session, fully attributed.
    assert "generation" in {span.attributes.get(OBSERVATION_TYPE) for span in spans}
    for span in spans:
        assert span.attributes.get(SESSION) == "thread-1", span.name
        assert span.attributes.get(USER) == str(job.user_id), span.name


@pytest.mark.asyncio
async def test_without_langfuse_a_research_job_gets_the_plain_config(monkeypatch) -> None:
    monkeypatch.setattr(core_tracing, "_hook", None)
    configs: list[RunnableConfig] = []
    job = _research_job()

    class _Specialist:
        async def aget_state(self, config: RunnableConfig):
            configs.append(config)
            return SimpleNamespace(created_at=None, next=(), values={})

        async def ainvoke(self, _input, *, config: RunnableConfig, **_kwargs):
            configs.append(config)
            return {}

    await runner._invoke(_Specialist(), job)

    assert configs == [{"configurable": {"thread_id": f"research:{job.id}"}}] * 2


@pytest.mark.asyncio
async def test_research_jobs_never_inherit_the_context_that_started_them(
    langfuse_capture, monkeypatch
) -> None:
    """Whatever is active where the poller starts, a job starts with nothing.

    A job inheriting a chat's LangChain parent run (which LangSmith's tracer
    follows) or its OpenTelemetry span (which Langfuse follows) would be filed
    under a tool call that ended long before the job did.
    """

    job_id = uuid4()
    started = asyncio.Event()
    inside_job: dict[str, object] = {}
    inside_chat: dict[str, object] = {}

    async def one_active_job():
        return [job_id]

    async def record_context(_job_id):
        inside_job["span"] = otel_trace.get_current_span().get_span_context().is_valid
        inside_job["parent_run"] = var_child_runnable_config.get()
        started.set()

    monkeypatch.setattr(runner.jobs, "active_ids", one_active_job)
    monkeypatch.setattr(runner, "_run", record_context)

    async def chat_turn(_input):
        # There is something to inherit here -- otherwise this test proves nothing.
        inside_chat["span"] = otel_trace.get_current_span().get_span_context().is_valid
        inside_chat["parent_run"] = var_child_runnable_config.get() is not None
        async with runner.lifespan():
            await asyncio.wait_for(started.wait(), timeout=5)

    with langfuse_capture.client.start_as_current_observation(name="chat turn"):
        await RunnableLambda(chat_turn).ainvoke("go")

    assert inside_chat == {"span": True, "parent_run": True}
    assert inside_job == {"span": False, "parent_run": None}


# --------------------------------------------------------------------------- #
# Shutdown: what was still running is flushed, not lost
# --------------------------------------------------------------------------- #


class _SlowModel(GenericFakeChatModel):
    """A model still mid-answer when the server is told to stop."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for number in range(200):
            await asyncio.sleep(0.05)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=f"word{number} "))
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def _slow_graph():
    model = _SlowModel(messages=iter([AIMessage(content="unused")]))

    async def agent(state: MessagesState):
        # `ainvoke`, as LangChain's agents call their model.
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


@pytest.fixture
def sse_not_exiting(monkeypatch):
    """sse-starlette's process-wide "server is exiting" flag, reset afterwards.

    Stopping a uvicorn sets it, and nothing clears it: every later SSE response
    in this process would end the moment it started.
    """

    monkeypatch.setattr(AppStatus, "should_exit", False)


def _graph_events(graph, run):
    """The agent's events, streamed the way core's `_stream_agent_payloads` does."""

    async def events():
        async for _ in graph.astream(
            _hello(),
            config=run.config,
            stream_mode=["updates", "messages", "custom"],
            subgraphs=True,
            version="v2",
        ):
            yield {"data": "event"}

    return events()


@pytest.mark.asyncio
async def test_stopping_the_server_mid_stream_flushes_the_graph_spans(
    langfuse_capture, sse_not_exiting
) -> None:
    """A real uvicorn and sse-starlette, stopped while a chat is streaming.

    sse-starlette ends the stream as soon as shutdown begins. Unless the graph
    has finished cancelling its nodes by then, their spans end after the
    Langfuse lifespan has flushed, and are lost.
    """

    graph = _slow_graph()

    @asynccontextmanager
    async def lifespan(_app):
        async with langfuse_tracing.langfuse_lifespan():
            yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/stream")
    async def stream():
        run = AgentInputHandler.build_run_context(thread_id="thread-1", user_id="user-1")
        return AgentStreamResponse(_graph_events(graph, run), on_close=lambda: None)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", ws="none")
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]

    async def chat() -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", f"http://127.0.0.1:{port}/stream") as response:
                async for _ in response.aiter_lines():
                    pass

    chatting = asyncio.create_task(chat())
    await asyncio.sleep(0.5)  # the model is now mid-answer
    # What uvicorn's SIGTERM handler sets. Calling the handler itself would
    # make uvicorn re-raise SIGTERM at this test process once it has stopped.
    server.should_exit = True
    await serving  # includes the lifespan shutdown, i.e. the flush
    await asyncio.gather(chatting, return_exceptions=True)

    exported = {span.name for span in langfuse_capture.exporter.get_finished_spans()}
    assert {"LangGraph", "agent"} <= exported
    # No node outlived its request.
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.asyncio
async def test_a_client_that_stops_reading_still_gets_its_graph_spans_flushed(
    langfuse_capture, sse_not_exiting
) -> None:
    """Cancelled while an event is being sent, not while waiting for one.

    The response is driven at the ASGI level so the send can be held open: a
    client that stopped reading. The generator feeding the response is then
    paused at its `yield`, where no `finally` of its own would ever run.
    """

    run = AgentInputHandler.build_run_context(thread_id="thread-1", user_id="user-1")
    released: list[bool] = []
    response = AgentStreamResponse(
        _graph_events(_slow_graph(), run), on_close=lambda: released.append(True)
    )
    client_left = asyncio.Event()
    first_event_sent = asyncio.Event()

    async def receive():
        await client_left.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_event_sent.set()
            await asyncio.Event().wait()  # never read

    serving = asyncio.create_task(response({"type": "http"}, receive, send))
    await first_event_sent.wait()
    client_left.set()
    await asyncio.wait_for(serving, timeout=5)
    async with langfuse_tracing.langfuse_lifespan():
        pass  # what the server does next, eventually: flush

    exported = {span.name for span in langfuse_capture.exporter.get_finished_spans()}
    assert {"LangGraph", "agent"} <= exported
    assert released == [True]
