"""Opt-in integration tests for the clean Postgres persistence foundation."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
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
from juena.server.chat.repository import (
    ChatNotFoundError,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
    load_thread_messages,
)
from juena.server.api.endpoints import _authorize_thread, router as api_router
from juena.server.chat.endpoints import router as chat_router
from juena.server.database.checkpointer import checkpointer_lifespan
from juena.server.database import database_lifespan, get_session_factory
from juena.server.auth.saml import PERSISTENT_NAME_ID_FORMAT, SamlIdentity
from juena.server.database.store import store_lifespan
from juena.server.database.models import AuthSession, utc_now
from juena.sandbox.jobs import SandboxJobs, hold_worker_lock
from juena.sandbox.worker import RESTART_MESSAGE

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
        assert {
            "users",
            "auth_sessions",
            "saml_login_requests",
            "chats",
            "sandbox_jobs",
        } <= set(table_names)
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


def _clear_sandbox_jobs(jobs: "SandboxJobs") -> None:
    """`sandbox_jobs` is transient metadata, so tests may start from empty."""

    with jobs.pool.connection() as connection:
        connection.execute("DELETE FROM sandbox_jobs")


@pytest.mark.asyncio
async def test_sandbox_admission_is_exact_tenant_scoped_and_leaves_no_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        jobs = SandboxJobs(TEST_DATABASE_URL, concurrency=4)
        await asyncio.to_thread(jobs.open)
        await asyncio.to_thread(_clear_sandbox_jobs, jobs)
        created = []

        async def submit(tenant: str, workspace: str):
            return await asyncio.to_thread(
                jobs.create,
                tenant_id=tenant,
                workspace_id=workspace,
                command="python -V",
                timeout_seconds=30,
                max_output_bytes=1000,
            )

        try:
            for index in range(4):
                job = await submit(f"{index:064x}", f"{index + 100:064x}")
                assert job is not None
                created.append(job)

            # The slot limit is exact: the fifth is busy, and no row is left
            # behind to hold a slot.
            assert await submit("f" * 64, "e" * 64) is None
            await asyncio.to_thread(
                jobs.finish,
                created[0].id,
                status="completed",
                output="done",
            )

            # A freed slot admits a new tenant but not the one already active.
            assert await submit(f"{1:064x}", f"{201:064x}") is None
            replacement = await submit("f" * 64, "e" * 64)
            assert replacement is not None
            created.append(replacement)
        finally:
            for job in created:
                if job is not None:
                    await asyncio.to_thread(
                        jobs.finish, job.id, status="failed", output="test cleanup"
                    )
            await asyncio.to_thread(jobs.close)


@pytest.mark.asyncio
async def test_concurrent_claims_never_select_the_same_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        jobs = SandboxJobs(TEST_DATABASE_URL, concurrency=4)
        await asyncio.to_thread(jobs.open)
        await asyncio.to_thread(_clear_sandbox_jobs, jobs)
        created = []
        try:
            for index in range(4):
                job = await asyncio.to_thread(
                    jobs.create,
                    tenant_id=f"{index + 300:064x}",
                    workspace_id=f"{index + 400:064x}",
                    command="python -V",
                    timeout_seconds=30,
                    max_output_bytes=1000,
                )
                assert job is not None
                created.append(job)

            claims = await asyncio.gather(
                *(asyncio.to_thread(jobs.claim) for _index in range(8))
            )
            claimed = [job for job in claims if job is not None]

            assert len(claimed) == 4
            assert len({job.id for job in claimed}) == 4
            assert all(job.status == "running" for job in claimed)
        finally:
            for job in created:
                await asyncio.to_thread(
                    jobs.finish, job.id, status="failed", output="test cleanup"
                )
            await asyncio.to_thread(jobs.close)


@pytest.mark.asyncio
async def test_workspace_guard_prevents_job_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        jobs = SandboxJobs(TEST_DATABASE_URL, concurrency=4)
        await asyncio.to_thread(jobs.open)
        await asyncio.to_thread(_clear_sandbox_jobs, jobs)
        workspace_id = f"{501:064x}"
        try:
            with jobs.workspace_guard(workspace_id) as available:
                assert available is True
                admitted = jobs.create(
                    tenant_id=f"{500:064x}",
                    workspace_id=workspace_id,
                    command="python -V",
                    timeout_seconds=30,
                    max_output_bytes=1000,
                )
                assert admitted is None
        finally:
            await asyncio.to_thread(jobs.close)


@pytest.mark.asyncio
async def test_stale_jobs_are_not_claimed_and_restart_fails_all_active_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        jobs = SandboxJobs(TEST_DATABASE_URL, concurrency=4)
        await asyncio.to_thread(jobs.open)
        await asyncio.to_thread(_clear_sandbox_jobs, jobs)
        try:
            stale = jobs.create(
                tenant_id=f"{600:064x}",
                workspace_id=f"{601:064x}",
                command="echo stale",
                timeout_seconds=30,
                max_output_bytes=1000,
            )
            assert stale is not None
            with jobs.pool.connection() as connection:
                connection.execute(
                    "UPDATE sandbox_jobs SET created_at = %s WHERE id = %s",
                    (datetime.now(UTC) - timedelta(minutes=1), stale.id),
                )
            assert jobs.claim() is None
            assert jobs.fail_stale_queued() == 1
            assert jobs.get(stale.id).status == "failed"  # type: ignore[union-attr]

            queued = jobs.create(
                tenant_id=f"{602:064x}",
                workspace_id=f"{603:064x}",
                command="echo queued",
                timeout_seconds=30,
                max_output_bytes=1000,
            )
            running = jobs.create(
                tenant_id=f"{604:064x}",
                workspace_id=f"{605:064x}",
                command="echo running",
                timeout_seconds=30,
                max_output_bytes=1000,
            )
            assert queued is not None and running is not None
            claimed = jobs.claim()
            assert claimed is not None
            assert jobs.fail_queued(claimed.id, "too late") is None
            assert jobs.get(claimed.id).status == "running"  # type: ignore[union-attr]
            assert jobs.fail_active(RESTART_MESSAGE) == 2
            assert jobs.get(queued.id).status == "failed"  # type: ignore[union-attr]
            assert jobs.get(running.id).status == "failed"  # type: ignore[union-attr]
        finally:
            await asyncio.to_thread(jobs.close)


@pytest.mark.asyncio
async def test_worker_lock_is_exclusive_and_visible_to_the_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        jobs = SandboxJobs(TEST_DATABASE_URL, concurrency=4)
        await asyncio.to_thread(jobs.open)
        try:
            if await asyncio.to_thread(jobs.worker_is_alive):
                pytest.skip("a sandbox worker is running; stop it to test the lock")

            def hold_and_probe() -> tuple[bool, bool, bool]:
                with hold_worker_lock(TEST_DATABASE_URL) as acquired:
                    with hold_worker_lock(TEST_DATABASE_URL) as second:
                        return acquired, second, jobs.worker_is_alive()

            acquired, second, visible = await asyncio.to_thread(hold_and_probe)

            assert acquired is True
            # A second worker exits instead of competing for jobs.
            assert second is False
            # Reading pg_locks never acquires, so a starting worker is unaffected.
            assert visible is True
            assert await asyncio.to_thread(jobs.worker_is_alive) is False
        finally:
            await asyncio.to_thread(jobs.close)


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
                    "/stream",
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
