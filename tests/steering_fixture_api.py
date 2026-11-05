"""In-memory API fixture for browser acceptance; never uses provider credentials."""
import asyncio
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import uvicorn
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from juena_core.agents.steering import SteeringMiddleware
from juena_core.agents.ask_user import build_ask_user_tool
from juena_core.artifacts import ArtifactStore, set_artifact_store_for_tests
from juena_core.server.api import endpoints
from juena_core.server.database.connection import get_db_session
from juena_core.sandbox.approvals import SandboxResumeInput, register_sandbox_interrupt


def start_fixture_api():
    @tool
    async def search(delay: int) -> str:
        """A deliberately slow, deterministic research step."""
        await asyncio.sleep(delay)
        return "Fixture evidence"

    @tool
    def execute(command: str) -> str:
        """Return a fixture result after approval; never execute a command."""
        return "Fixture command result"

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            last_user = next(m for m in reversed(messages) if isinstance(m, HumanMessage) and not m.additional_kwargs.get("steering"))
            after = messages[messages.index(last_user)+1:]
            results = [m.tool_call_id for m in after if isinstance(m, ToolMessage)]
            if not results:
                call = {"name":"search","args":{"delay":600 if "slow" in last_user.content else 5},"id":"fixture-search"}
                answer = AIMessage(content="", tool_calls=[call])
            elif "ask" in last_user.content and "fixture-question" not in results:
                call = {"name":"ask_user","args":{"question":"Which fixture scope?","options":["FAIR","NeXus"]},"id":"fixture-question"}
                answer = AIMessage(content="", tool_calls=[call])
            elif "approval" in last_user.content and "fixture-command" not in results:
                call = {"name":"execute","args":{"command":"printf fixture"},"id":"fixture-command"}
                answer = AIMessage(content="", tool_calls=[call])
            else:
                correction = next((m.content for m in reversed(after) if isinstance(m, HumanMessage)), "original scope")
                answer = AIMessage(content=f"Fixture answer: {correction}")
            return ChatResult(generations=[ChatGeneration(message=answer)])

    graph = create_agent(
        Model(responses=[AIMessage(content="unused")]),
        tools=[search, execute, build_ask_user_tool("fixture")],
        middleware=[SteeringMiddleware("supervisor"), HumanInTheLoopMiddleware(interrupt_on={"execute": True})],
        checkpointer=InMemorySaver(),
    )

    async def agent(*args, **kwargs):
        return graph

    async def owned(session, user_id, thread_id, agent_id=None, **kwargs):
        return SimpleNamespace(thread_id=thread_id, updated_at=None)

    async def prepare(*args, **kwargs):
        return None

    class Session:
        async def commit(self):
            pass

    async def session():
        yield Session()

    async def principal():
        return SimpleNamespace(id=UUID(int=1))

    endpoints.get_agent = agent
    endpoints.ensure_owned_chat = owned
    endpoints.get_owned_chat = owned
    endpoints.prepare_code_chat_turn_inputs = prepare
    root = Path(tempfile.mkdtemp(prefix="juena-steering-browser-"))
    set_artifact_store_for_tests(ArtifactStore(root / "artifacts", root / "audit.jsonl"))
    app = FastAPI()
    app.dependency_overrides[get_db_session] = session
    register_sandbox_interrupt()
    app.include_router(endpoints.build_api_router(principal, run_controls=True, resume_input=SandboxResumeInput))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=19622, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 5
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Fixture API did not start")
        time.sleep(0.02)
    return server

