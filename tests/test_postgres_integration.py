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
from sqlalchemy import inspect, select

from juena_core.agents.findings import validate_findings
import dataclasses

import juena_core
import juena_core.config
from juena_core.sandbox import config as sandbox_config

# Imported for its side effect, and only here: `sandbox_jobs` joins
# `Base.metadata` when an application enables the sandbox, which is what keeps
# the table out of a deployment that runs no worker. Half of this module
# exercises that queue against a real database, so it needs the table.
import juena_core.sandbox.models  # noqa: F401  attaches sandbox_jobs

from juena.agents.juena_agent import JUENA_AGENT_ID
from juena.core.config import global_config
from fastapi import HTTPException
from juena.server.auth import (
    Principal,
    create_auth_session,
    session_principal,
    hash_session_token,
    revoke_auth_session,
    upsert_saml_user,
)
from juena_core.server.chat.repository import (
    ChatNotFoundError,
    ensure_owned_chat,
    get_owned_chat,
    list_owned_chats,
    load_thread_messages,
)
from juena_core.server.api.endpoints import build_api_router
from juena_core.server.chat.endpoints import build_chat_router
from juena_core.server.database.checkpointer import checkpointer_lifespan
from juena_core.server.database.connection import database_lifespan, get_session_factory
from juena.server.auth.saml import PERSISTENT_NAME_ID_FORMAT, SamlIdentity
from juena_core.server.database.store import store_lifespan
from juena_core.server.database.models import AuthSession, User, utc_now
from juena.research import jobs as research_jobs
from juena_core.sandbox.jobs import SandboxJobs, hold_worker_lock
from juena_core.sandbox.worker import RESTART_MESSAGE

TEST_DATABASE_URL = os.getenv("JUENA_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set JUENA_TEST_DATABASE_URL to run Postgres integration tests",
)


@pytest.fixture(autouse=True)
def _use_test_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point *both* configurations at the throwaway database.

    There are two of them since the cutover: this application's `Config`, which
    its own modules read, and juena_core's frozen `CoreSettings`, which is what
    `connection.py` actually opens a pool against. Core does not read
    `global_config` and never will -- that is the whole point of decision 1 --
    so patching only the application's would put this suite back on the
    developer's database, the exact regression commit `40c846b` fixed.

    Autouse, because the failure is silent: the tests still pass, against the
    wrong database, destroying rows someone was using.
    """

    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(
        juena_core.config,
        "_settings",
        dataclasses.replace(juena_core.settings(), DATABASE_URL=TEST_DATABASE_URL),
    )
    assert juena_core.settings().DATABASE_URL == TEST_DATABASE_URL

    # The sandbox is opt-in core now, and `conftest.py` turns it off for the
    # unit suite. This module is one of the places that turns it back on
    # explicitly: half of it exercises the job queue.
    monkeypatch.setattr(
        sandbox_config,
        "_sandbox_settings",
        dataclasses.replace(
            sandbox_config.sandbox_settings(),
            enabled=True,
            identity_secret="integration-test-secret-of-sufficient-length",
        ),
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
            "research_jobs",
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
            chat = await ensure_owned_chat(session, alice.id, f"thread-{uuid4()}", JUENA_AGENT_ID)
            await session.commit()

            assert (
                await get_owned_chat(
                    session, alice.id, chat.thread_id, agent_id=JUENA_AGENT_ID
                )
            ).user_id == alice.id
            with pytest.raises(ChatNotFoundError):
                await get_owned_chat(
                    session, bob.id, chat.thread_id, agent_id=JUENA_AGENT_ID
                )

            newer_chat = await ensure_owned_chat(
                session,
                alice.id,
                f"thread-{uuid4()}",
                JUENA_AGENT_ID,
            )
            chat.updated_at = utc_now() - timedelta(days=2)
            newer_chat.updated_at = utc_now() - timedelta(days=1)
            await session.commit()
            assert (await list_owned_chats(session, alice.id))[0].thread_id == newer_chat.thread_id

            # The recency half of this test moved to juena-core, where
            # `_authorize_thread` now lives as a closure inside
            # `build_api_router` rather than a module-level function this could
            # call. Core asserts it through the real route, against a real
            # database, in `test_server_routes_postgres.py`.

            token = await create_auth_session(session, alice)
            await session.commit()
            stored_session = await session.get(AuthSession, hash_session_token(token))
            assert stored_session is not None
            assert stored_session.token_hash != token
            assert (await session_principal(session_token=token, session=session)).id == alice.id

            await revoke_auth_session(session, token)
            await session.commit()
            with pytest.raises(HTTPException) as revoked:
                await session_principal(session_token=token, session=session)
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
                await session_principal(session_token=expired_token, session=session)
            assert expired.value.status_code == 401


def _clear_sandbox_jobs(jobs: "SandboxJobs") -> None:
    """`sandbox_jobs` is transient metadata, so tests may start from empty."""

    with jobs.pool.connection() as connection:
        connection.execute("DELETE FROM sandbox_jobs")


@pytest.mark.asyncio
async def test_sandbox_admission_is_exact_tenant_scoped_and_leaves_no_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "alice": alice_model,
            "bob": bob_model,
        }
        active = {"name": "alice"}

        async def authenticated_user() -> Principal:
            return users[active["name"]]

        # Core's routers are factories over the identity dependency now, so
        # the dependency is visible in the signature rather than assumed. The
        # override still keys on the same callable object.
        app = FastAPI()
        app.include_router(build_chat_router(session_principal))
        app.include_router(build_api_router(session_principal))
        app.dependency_overrides[session_principal] = authenticated_user
        transport = httpx.ASGITransport(app=app)
        thread_id = f"owned-{uuid4()}"
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/chats",
                # `agent_id` is required on the wire now: a conversation that
                # does not know its own graph is the bug decision 13 exists to
                # prevent, and this application has exactly one answer.
                json={
                    "thread_id": thread_id,
                    "agent_id": JUENA_AGENT_ID,
                    "title": "Alice chat",
                },
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
                await ensure_owned_chat(session, owner.id, thread_id, JUENA_AGENT_ID)
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


# --------------------------------------------------------------------------- #
# Background research jobs
# --------------------------------------------------------------------------- #


async def _research_user(prefix: str):
    """One authenticated user, committed, for a research test."""
    factory = get_session_factory()
    async with factory() as session:
        user = await upsert_saml_user(
            session,
            SamlIdentity(
                subject=f"{prefix}-{uuid4()}",
                subject_format=PERSISTENT_NAME_ID_FORMAT,
                issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
                in_response_to=f"request-{prefix}",
            ),
        )
        await session.commit()
        return user.id


async def _create_research(**kwargs):
    return await research_jobs.create(
        run_input={"provider": "blablador", "model": "m", "files": {}}, **kwargs,
    )


async def _finish_research(job_id, **kwargs):
    async with research_jobs.claim(job_id) as claimed:
        assert claimed is not None
        await research_jobs.finish(claimed[1], job_id, **kwargs)


@pytest.mark.asyncio
async def test_research_concurrency_is_per_user_and_counts_only_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two in flight is a spend limit as much as a concurrency limit.

    Each background job is a full specialist loop against the model provider, so
    an agent that can start them freely can also spend freely.
    """
    async with database_lifespan():
        alice = await _research_user("research-alice")
        bob = await _research_user("research-bob")
        thread = f"thread-{uuid4()}"

        first = await _create_research(
            user_id=alice, thread_id=thread, specialist="s", objective="one", concurrency=2
        )
        second = await _create_research(
            user_id=alice, thread_id=thread, specialist="s", objective="two", concurrency=2
        )
        third = await _create_research(
            user_id=alice, thread_id=thread, specialist="s", objective="three", concurrency=2
        )
        assert first is not None and second is not None
        assert third is None, "the third job must be refused, not queued"

        # Another user is not affected by Alice's spend.
        assert await _create_research(
            user_id=bob, thread_id=thread, specialist="s", objective="bob's", concurrency=2
        ) is not None

        # Finishing one frees the slot.
        await _finish_research(first.id, status="succeeded", report="done")
        assert await _create_research(
            user_id=alice, thread_id=thread, specialist="s", objective="four", concurrency=2
        ) is not None


@pytest.mark.asyncio
async def test_admission_waits_on_the_owner_row_not_just_a_shared_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two starts arriving together must not both see the last free slot.

    Sharing a transaction is not enough. Under PostgreSQL's default Read
    Committed isolation a `COUNT` does not block a concurrent `INSERT`, so both
    callers could count the same free slot and both admit. `create` therefore
    locks the owner's `users` row before counting.

    Asserted by *blocking*, not by racing. A plain `asyncio.gather` of two
    `create` calls is timing-dependent -- whether both count before either
    inserts depends on how the driver happens to yield -- so it can pass against
    an unlocked build by accident, and a test that only sometimes catches the
    bug reads as proof the bug is gone.

    **The held lock must be `FOR KEY SHARE`, not `FOR UPDATE`.** An earlier
    version of this test held `FOR UPDATE` and passed against an unlocked build,
    because inserting a `research_jobs` row takes `FOR KEY SHARE` on the
    referenced `users` row for the foreign key -- and that conflicts with a held
    `FOR UPDATE`. Admission blocked either way and the test proved nothing.
    `FOR KEY SHARE` is compatible with the foreign key's own lock and conflicts
    only with `FOR UPDATE`, so it blocks the explicit lock and nothing else.
    """
    async with database_lifespan():
        alice = await _research_user("research-lock-alice")
        bob = await _research_user("research-lock-bob")
        thread = f"thread-{uuid4()}"

        factory = get_session_factory()
        async with factory() as holder:
            async with holder.begin():
                # FOR KEY SHARE: conflicts with `create`'s FOR UPDATE, and with
                # nothing else this transaction does. See the docstring.
                await holder.execute(
                    select(User.id)
                    .where(User.id == alice)
                    .with_for_update(read=True, key_share=True)
                )

                admitted = asyncio.create_task(
                    _create_research(
                        user_id=alice,
                        thread_id=thread,
                        specialist="s",
                        objective="waits on the owner row",
                        concurrency=2,
                    )
                )

                # Control: a different owner is not blocked. This also proves the
                # wait below is the row lock rather than an exhausted connection
                # pool, which would stall every caller equally.
                assert await asyncio.wait_for(
                    _create_research(
                        user_id=bob,
                        thread_id=thread,
                        specialist="s",
                        objective="different owner, never waits",
                        concurrency=2,
                    ),
                    timeout=10.0,
                ) is not None

                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(admitted), timeout=1.0)
            # The holder's transaction commits here, releasing Alice's row.

        job = await asyncio.wait_for(admitted, timeout=10.0)
        assert job is not None, "admission must proceed once the owner row is free"


@pytest.mark.asyncio
async def test_research_claim_excludes_other_workers_and_is_recoverable() -> None:
    async with database_lifespan():
        user = await _research_user("research-recovery")
        job = await _create_research(
            user_id=user, thread_id=f"thread-{uuid4()}", specialist="s",
            objective="interrupted", concurrency=4,
        )
        async with research_jobs.claim(job.id) as first:
            assert first is not None
            started_at = first[0].started_at
            async with research_jobs.claim(job.id) as competing:
                assert competing is None
            assert (await research_jobs.get(job.id, user)).status == "running"
        # The connection closes without a terminal write, just as on shutdown.
        assert job.id in await research_jobs.active_ids()
        async with research_jobs.claim(job.id) as recovered:
            assert recovered is not None
            assert recovered[0].started_at == started_at
            await research_jobs.finish(recovered[1], job.id, status="succeeded", report="recovered")
        assert job.id not in await research_jobs.active_ids()
        async with research_jobs.claim(job.id) as completed:
            assert completed is None


@pytest.mark.asyncio
async def test_a_collected_result_is_never_reported_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`list_research` answers "what do I still owe the user?".

    Without `collected_at` the supervisor would re-report the same finished job
    on every following turn.
    """
    async with database_lifespan():
        user = await _research_user("research-collect")
        thread = f"thread-{uuid4()}"

        job = await _create_research(
            user_id=user, thread_id=thread, specialist="s", objective="survey", concurrency=4
        )
        assert job is not None
        await _finish_research(
            job.id,
            status="succeeded",
            report="<specialist_report>\nStatus: OK\n</specialist_report>",
            findings={"/findings/survey.md": {"content": "paths and line numbers"}},
        )

        pending = await research_jobs.for_thread(thread, user)
        assert [j.id for j in pending] == [job.id]
        assert pending[0].findings == {
            "/findings/survey.md": {"content": "paths and line numbers"}
        }

        await research_jobs.mark_collected(job.id)

        assert await research_jobs.for_thread(thread, user) == []
        # Still readable by id -- collected means "already reported", not "gone".
        assert (await research_jobs.get(job.id, user)) is not None


@pytest.mark.asyncio
async def test_bulk_acknowledgement_stamps_once_and_leaves_other_jobs_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mark_collected_many` runs at the end of every supervisor turn.

    It is handed the whole delivery marker each time, which grows for the life of
    the conversation, so it has to be idempotent -- re-stamping an already
    collected job would move `collected_at` forward on every turn -- and it must
    touch nothing outside the ids it was given.
    """
    async with database_lifespan():
        user = await _research_user("research-bulk-collect")
        thread = f"thread-{uuid4()}"

        async def _finished(objective: str):  # noqa: ANN202
            job = await _create_research(
                user_id=user,
                thread_id=thread,
                specialist="s",
                objective=objective,
                concurrency=4,
            )
            assert job is not None
            await _finish_research(job.id, status="succeeded", report="done")
            return job

        delivered = await _finished("the one the conversation received")
        untouched = await _finished("still waiting to be collected")

        assert await research_jobs.mark_collected_many([delivered.id]) == 1

        first_stamp = (await research_jobs.get(delivered.id, user)).collected_at
        assert first_stamp is not None

        # The next turn passes the same marker again. Nothing left to do, and the
        # original timestamp must survive.
        assert await research_jobs.mark_collected_many([delivered.id]) == 0
        assert (await research_jobs.get(delivered.id, user)).collected_at == first_stamp

        assert (await research_jobs.get(untouched.id, user)).collected_at is None
        assert [j.id for j in await research_jobs.for_thread(thread, user)] == [untouched.id]


@pytest.mark.asyncio
async def test_findings_survive_postgres_with_their_full_file_data_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The column is JSON, so nothing on the database side guarantees the shape.

    `findings` was annotated `dict[str, str]` while it actually held Deep Agents
    `FileData`, and the value is merged straight back into the graph's `files`
    channel on collection -- where a malformed entry breaks every later read of
    that path. This exercises the real round trip rather than the annotation:
    every `FileData` key must come back, and `base_findings` must come back
    alongside it, because a conflict cannot be detected without the revision the
    job read.
    """
    async with database_lifespan():
        user = await _research_user("research-shape")
        thread = f"thread-{uuid4()}"

        stored = {
            "/findings/survey.md": {
                "content": "silx fits with leastsq.py line 40",
                "encoding": "utf-8",
                "created_at": "2026-08-19T10:00:00+00:00",
                "modified_at": "2026-08-19T10:05:00+00:00",
            }
        }
        job = await _create_research(
            user_id=user, thread_id=thread, specialist="s", objective="survey", concurrency=4
        )
        assert job is not None
        await _finish_research(
            job.id,
            status="succeeded",
            report="<specialist_report>\nStatus: OK\n</specialist_report>",
            findings=stored,
            base_findings={"/findings/survey.md": "what the job read"},
        )

        fetched = await research_jobs.get(job.id, user)
        assert fetched is not None
        assert fetched.findings == stored, "a FileData key was lost in the round trip"
        assert fetched.base_findings == {"/findings/survey.md": "what the job read"}
        # And what actually reaches graph state is the validated form.
        assert validate_findings(fetched.findings) == stored


@pytest.mark.asyncio
async def test_one_users_research_is_invisible_to_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with database_lifespan():
        alice = await _research_user("research-owner")
        bob = await _research_user("research-intruder")
        thread = f"thread-{uuid4()}"

        job = await _create_research(
            user_id=alice, thread_id=thread, specialist="s", objective="private", concurrency=4
        )
        assert job is not None

        assert await research_jobs.get(job.id, bob) is None
        assert await research_jobs.for_thread(thread, bob) == []


async def _research_recovery_process(job_id: str, evidence_path: str, pause: str) -> None:
    """Subprocess probe: real runner/checkpoints, deterministic research nodes."""
    from pathlib import Path
    from types import SimpleNamespace
    from uuid import UUID
    from juena.research import runner
    from juena_core.agents.specialist_outcome import SpecialistOutcomeMiddleware
    from juena_core.server.agent import registry

    juena_core.config._settings = dataclasses.replace(
        juena_core.settings(), DATABASE_URL=TEST_DATABASE_URL,
    )

    class State(MessagesState):
        files: dict

    async def wait_for_crash():
        print("RESEARCH_WAITING", flush=True)
        await asyncio.Event().wait()

    def build(*, checkpointer):
        def search(state):
            with Path(evidence_path).open("a") as output:
                output.write("completed search\n")
            assert state["files"]["/findings/base.md"]["content"] == "original input"
            return {"files": {
                **state["files"],
                "/findings/recovered.md": create_file_data("saved evidence"),
            }}

        async def report(state):
            if pause == "mid-step":
                await wait_for_crash()
            assert "/findings/recovered.md" in state["files"]
            return SpecialistOutcomeMiddleware(specialist_name="software-specialist").after_agent(
                {"structured_response": {
                    "status": "completed", "finding": "Recovered survey",
                    "evidence": ["saved evidence"],
                }},
                SimpleNamespace(),
            )

        builder = StateGraph(State)
        builder.add_node("search", search)
        builder.add_node("report", report)
        builder.add_edge(START, "search")
        builder.add_edge("search", "report")
        builder.add_edge("report", END)
        return {"runnable": builder.compile(checkpointer=checkpointer)}

    async def get_agent(_name):
        return None

    registry.get_agent = get_agent
    registry.get_agent_resources = lambda _name: SimpleNamespace(
        background_specialists={"software-specialist": build},
    )
    if pause == "after-graph":
        async def finish(*args, **kwargs):
            assert kwargs["status"] == "succeeded", kwargs
            await wait_for_crash()
        research_jobs.finish = finish

    async with database_lifespan(), checkpointer_lifespan():
        await runner._run(UUID(job_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", ["mid-step", "after-graph"])
async def test_research_recovers_after_process_death_without_repeating_completed_work(
    tmp_path, pause,
) -> None:
    import sys

    evidence_path = tmp_path / "completed-searches.txt"
    async with database_lifespan():
        user = await _research_user("research-process-recovery")
        job = await research_jobs.create(
            user_id=user, thread_id=f"thread-{uuid4()}", specialist="software-specialist",
            objective="survey", concurrency=2,
            run_input={"provider": "blablador", "model": "m", "files": {
                "/findings/base.md": create_file_data("original input"),
            }},
        )
        script = (
            "import asyncio, sys; import tests.conftest; "
            "from tests.test_postgres_integration import _research_recovery_process; "
            "asyncio.run(_research_recovery_process(*sys.argv[1:]))"
        )

        async def spawn(mode):
            return await asyncio.create_subprocess_exec(
                sys.executable, "-c", script, str(job.id), str(evidence_path), mode,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )

        process = await spawn(pause)
        output = []
        try:
            async with asyncio.timeout(30):
                while True:
                    line = await process.stdout.readline()
                    assert line, b"".join(output).decode()
                    output.append(line)
                    if b"RESEARCH_WAITING" in line:
                        break
            before = await research_jobs.get(job.id, user)
            assert before.status == "running"
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()

        recovered_process = await spawn("resume")
        try:
            output, _ = await asyncio.wait_for(recovered_process.communicate(), 30)
            assert recovered_process.returncode == 0, output.decode()
        finally:
            if recovered_process.returncode is None:
                recovered_process.kill()
                await recovered_process.wait()
        recovered = await research_jobs.get(job.id, user)
        assert recovered.status == "succeeded", output.decode()
        assert recovered.started_at == before.started_at
        assert "Recovered survey" in recovered.report
        assert set(recovered.findings) == {"/findings/recovered.md"}
        assert recovered.base_findings == {"/findings/base.md": "original input"}
        assert evidence_path.read_text() == "completed search\n"


@pytest.mark.asyncio
async def test_a_lost_research_connection_cannot_overwrite_the_new_owner():
    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with database_lifespan(), checkpointer_lifespan():
        user = await _research_user("research-lost-connection")
        job = await _create_research(
            user_id=user, thread_id=f"thread-{uuid4()}", specialist="s",
            objective="survey", concurrency=2,
        )
        async with research_jobs.claim(job.id) as first:
            connection = first[1]
            graph = _counter_graph(AsyncPostgresSaver(connection))
            config = {"configurable": {"thread_id": f"research:{job.id}"}}
            await graph.ainvoke({"count": 0}, config, durability="sync")
            await connection.close()
            async with research_jobs.claim(job.id) as recovered:
                await research_jobs.finish(recovered[1], job.id, status="succeeded", report="new owner")
            with pytest.raises(psycopg.Error):
                await graph.ainvoke({"count": 99}, config, durability="sync")
            with pytest.raises(psycopg.Error):
                await research_jobs.finish(connection, job.id, status="failed", error="old owner")
        result = await research_jobs.get(job.id, user)
        assert result.status == "succeeded"
        assert result.report == "new owner"


@pytest.mark.asyncio
async def test_research_upgrade_preserves_reports_and_fails_only_legacy_active_jobs():
    from pathlib import Path
    from psycopg import AsyncConnection
    from juena_core.server.database.connection import psycopg_database_url

    # A session-local table shadows the real one. No application rows are changed.
    async with await AsyncConnection.connect(
        psycopg_database_url(TEST_DATABASE_URL), autocommit=True,
    ) as connection:
        await connection.execute(
            "CREATE TEMP TABLE research_jobs "
            "(id int, status text, error text, report text, finished_at timestamptz)"
        )
        await connection.execute(
            "INSERT INTO research_jobs (id, status, report) "
            "VALUES (1, 'running', ''), (2, 'succeeded', 'keep this report')"
        )
        migration = (Path(__file__).parents[1] / "deploy/research-recovery.sql").read_text()
        await connection.execute(migration)
        await connection.execute(
            "INSERT INTO research_jobs (id, status, run_input) "
            "VALUES (3, 'running', '{\"provider\":\"blablador\"}')"
        )
        await connection.execute(migration)
        cursor = await connection.execute(
            "SELECT id, status, report, error FROM research_jobs ORDER BY id"
        )
        old_active, completed, new_active = await cursor.fetchall()
        assert old_active[1] == "failed"
        assert "predates research recovery" in old_active[3]
        assert completed[1:3] == ("succeeded", "keep this report")
        assert new_active[1] == "running"
