"""Opt-in integration tests for the clean Postgres persistence foundation."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TypedDict
from uuid import uuid4

import pytest
import httpx
from fastapi import FastAPI
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, START, StateGraph
from sqlalchemy import inspect

from juena.core.config import global_config
from fastapi import HTTPException
from juena.server.auth import (
    AuthenticatedUser,
    create_auth_session,
    get_current_user,
    hash_session_token,
    revoke_auth_session,
    upsert_saml_user,
)
from juena.server.chat_repository import (
    ChatNotFoundError,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
    load_thread_messages,
)
from juena.server.api_endpoints import _authorize_thread, router as api_router
from juena.server.chat_endpoints import router as chat_router
from juena.server.checkpointer import checkpointer_lifespan
from juena.server.database import database_lifespan, get_session_factory
from juena.server.saml_service import PERSISTENT_NAME_ID_FORMAT, SamlIdentity
from juena.server.store import store_lifespan
from juena.server.models import AuthSession, utc_now

TEST_DATABASE_URL = os.getenv("JUENA_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set JUENA_TEST_DATABASE_URL to run Postgres integration tests",
)


class _CounterState(TypedDict):
    count: int


def _counter_graph(checkpointer):  # noqa: ANN001
    builder = StateGraph(_CounterState)
    builder.add_node("increment", lambda state: {"count": state["count"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_clean_schema_and_user_owned_chats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan() as engine:
        async with engine.connect() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names()
            )
        assert {"users", "auth_sessions", "saml_login_requests", "chats"} <= set(table_names)
        # Message content belongs to the checkpointer alone; a mirror table
        # would give the UI and the agent two histories that can drift.
        assert "chat_messages" not in table_names

        factory = get_session_factory()
        async with factory() as session:
            alice = await upsert_saml_user(
                session,
                SamlIdentity(
                    subject=f"alice-{uuid4()}",
                    subject_format=PERSISTENT_NAME_ID_FORMAT,
                    issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                    in_response_to="request-alice",
                ),
            )
            bob = await upsert_saml_user(
                session,
                SamlIdentity(
                    subject=f"bob-{uuid4()}",
                    subject_format=PERSISTENT_NAME_ID_FORMAT,
                    issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                    in_response_to="request-bob",
                ),
            )
            chat = await ensure_owned_chat(session, alice.id, f"thread-{uuid4()}")
            await session.commit()

            assert (await get_owned_chat(session, alice.id, chat.thread_id)).user_id == alice.id
            with pytest.raises(ChatNotFoundError):
                await get_owned_chat(session, bob.id, chat.thread_id)

            newer_chat = await ensure_owned_chat(
                session,
                alice.id,
                f"thread-{uuid4()}",
            )
            chat.updated_at = utc_now() - timedelta(days=2)
            newer_chat.updated_at = utc_now() - timedelta(days=1)
            await session.commit()
            assert (await list_owned_chats(session, alice.id))[0].thread_id == newer_chat.thread_id

            await _authorize_thread(
                session,
                AuthenticatedUser(alice.id, alice.saml_subject, None, None),
                chat.thread_id,
            )
            assert (await list_owned_chats(session, alice.id))[0].thread_id == chat.thread_id

            token = await create_auth_session(session, alice)
            await session.commit()
            stored_session = await session.get(AuthSession, hash_session_token(token))
            assert stored_session is not None
            assert stored_session.token_hash != token
            assert (await get_current_user(session_token=token, session=session)).id == alice.id

            await revoke_auth_session(session, token)
            await session.commit()
            with pytest.raises(HTTPException) as revoked:
                await get_current_user(session_token=token, session=session)
            assert revoked.value.status_code == 401

            expired_token = await create_auth_session(session, alice)
            expired_session = await session.get(
                AuthSession,
                hash_session_token(expired_token),
            )
            assert expired_session is not None
            expired_session.expires_at = utc_now() - timedelta(seconds=1)
            await session.commit()
            with pytest.raises(HTTPException) as expired:
                await get_current_user(session_token=expired_token, session=session)
            assert expired.value.status_code == 401


@pytest.mark.asyncio
async def test_chat_and_agent_endpoints_enforce_thread_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        factory = get_session_factory()
        async with factory() as session:
            alice_model = await upsert_saml_user(
                session,
                SamlIdentity(
                    subject=f"alice-api-{uuid4()}",
                    subject_format=PERSISTENT_NAME_ID_FORMAT,
                    issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                    in_response_to="request-alice-api",
                ),
            )
            bob_model = await upsert_saml_user(
                session,
                SamlIdentity(
                    subject=f"bob-api-{uuid4()}",
                    subject_format=PERSISTENT_NAME_ID_FORMAT,
                    issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                    in_response_to="request-bob-api",
                ),
            )
            await session.commit()

        users = {
            "alice": AuthenticatedUser(alice_model.id, alice_model.saml_subject, None, None),
            "bob": AuthenticatedUser(bob_model.id, bob_model.saml_subject, None, None),
        }
        active = {"name": "alice"}

        async def authenticated_user() -> AuthenticatedUser:
            return users[active["name"]]

        app = FastAPI()
        app.include_router(chat_router)
        app.include_router(api_router)
        app.dependency_overrides[get_current_user] = authenticated_user
        transport = httpx.ASGITransport(app=app)
        thread_id = f"owned-{uuid4()}"
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/chats",
                json={"thread_id": thread_id, "title": "Alice chat"},
            )
            assert created.status_code == 201

            active["name"] = "bob"
            assert (await client.get(f"/chats/{thread_id}")).status_code == 404
            assert (
                await client.patch(f"/chats/{thread_id}", json={"title": "stolen"})
            ).status_code == 404
            assert (await client.delete(f"/threads/{thread_id}")).status_code == 404
            assert (
                await client.post(
                    "/react_agent/invoke",
                    json={"message": "resume", "thread_id": thread_id},
                )
            ).status_code == 404
            assert (await client.get("/chats")).json() == []


@pytest.mark.asyncio
async def test_chat_history_is_served_from_the_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/chats/{id}` reads the checkpointer, so no second message store is needed."""

    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    thread_id = f"history-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async with database_lifespan():
        async with checkpointer_lifespan() as saver:
            builder = StateGraph(MessagesState)
            builder.add_node(
                "reply",
                lambda state: {"messages": [AIMessage(content="pong")]},
            )
            builder.add_edge(START, "reply")
            builder.add_edge("reply", END)
            graph = builder.compile(checkpointer=saver)
            await graph.ainvoke(
                {"messages": [HumanMessage(content="ping")]},
                config=config,
            )

            factory = get_session_factory()
            async with factory() as session:
                owner = await upsert_saml_user(
                    session,
                    SamlIdentity(
                        subject=f"history-{uuid4()}",
                        subject_format=PERSISTENT_NAME_ID_FORMAT,
                        issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                        in_response_to="request-history",
                    ),
                )
                await ensure_owned_chat(session, owner.id, thread_id)
                await session.commit()

            messages = await load_thread_messages(thread_id)

        assert [(message.type, message.content) for message in messages] == [
            ("human", "ping"),
            ("ai", "pong"),
        ]


@pytest.mark.asyncio
async def test_checkpoint_and_agents_file_survive_resource_recreation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    thread_id = f"thread-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    alice = str(uuid4())
    bob = str(uuid4())

    async with database_lifespan():
        async with checkpointer_lifespan() as saver:
            graph = _counter_graph(saver)
            assert (await graph.ainvoke({"count": 1}, config=config))["count"] == 2

        async with checkpointer_lifespan() as saver:
            graph = _counter_graph(saver)
            assert (await graph.aget_state(config)).values["count"] == 2

        async with store_lifespan() as store:
            await store.aput(
                ("memories", alice),
                "/AGENTS.md",
                create_file_data("# Preferences\n\n- Prefer Python examples.\n"),
                index=False,
            )

        async with store_lifespan() as store:
            alice_memory = await store.aget(("memories", alice), "/AGENTS.md")
            bob_memory = await store.aget(("memories", bob), "/AGENTS.md")
            assert alice_memory is not None
            assert "Prefer Python" in alice_memory.value["content"]
            assert bob_memory is None
