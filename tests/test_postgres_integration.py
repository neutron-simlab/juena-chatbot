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

from juena.agents.findings import validate_findings
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
from juena.server.database.models import AuthSession, User, utc_now
from juena.research import jobs as research_jobs
from juena.research.constants import RESTART_MESSAGE as RESEARCH_RESTART_MESSAGE
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


@pytest.mark.asyncio
async def test_research_concurrency_is_per_user_and_counts_only_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two in flight is a spend limit as much as a concurrency limit.

    Each background job is a full specialist loop against the model provider, so
    an agent that can start them freely can also spend freely.
    """
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        alice = await _research_user("research-alice")
        bob = await _research_user("research-bob")
        thread = f"thread-{uuid4()}"

        first = await research_jobs.create(
            user_id=alice, thread_id=thread, specialist="s", objective="one", concurrency=2
        )
        second = await research_jobs.create(
            user_id=alice, thread_id=thread, specialist="s", objective="two", concurrency=2
        )
        third = await research_jobs.create(
            user_id=alice, thread_id=thread, specialist="s", objective="three", concurrency=2
        )
        assert first is not None and second is not None
        assert third is None, "the third job must be refused, not queued"

        # Another user is not affected by Alice's spend.
        assert await research_jobs.create(
            user_id=bob, thread_id=thread, specialist="s", objective="bob's", concurrency=2
        ) is not None

        # Finishing one frees the slot.
        await research_jobs.finish(first.id, status="succeeded", report="done")
        assert await research_jobs.create(
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
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
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
                    research_jobs.create(
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
                    research_jobs.create(
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
async def test_a_job_left_running_by_a_dead_process_is_failed_not_left_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run lived in that process's event loop and did not survive it.

    A row still reading `running` would make `check_research` say "still working"
    forever, which is the one answer that is never true.
    """
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        user = await _research_user("research-orphan")
        thread = f"thread-{uuid4()}"

        job = await research_jobs.create(
            user_id=user, thread_id=thread, specialist="s", objective="interrupted", concurrency=4
        )
        assert job is not None
        await research_jobs.mark_running(job.id)

        await research_jobs.fail_orphans()

        after = await research_jobs.get(job.id, user)
        assert after is not None
        assert after.status == "failed"
        assert after.error == RESEARCH_RESTART_MESSAGE
        assert after.finished_at is not None


@pytest.mark.asyncio
async def test_a_collected_result_is_never_reported_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`list_research` answers "what do I still owe the user?".

    Without `collected_at` the supervisor would re-report the same finished job
    on every following turn.
    """
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        user = await _research_user("research-collect")
        thread = f"thread-{uuid4()}"

        job = await research_jobs.create(
            user_id=user, thread_id=thread, specialist="s", objective="survey", concurrency=4
        )
        assert job is not None
        await research_jobs.finish(
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
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
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
        job = await research_jobs.create(
            user_id=user, thread_id=thread, specialist="s", objective="survey", concurrency=4
        )
        assert job is not None
        await research_jobs.finish(
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
    monkeypatch.setattr(global_config, "DATABASE_URL", TEST_DATABASE_URL)
    async with database_lifespan():
        alice = await _research_user("research-owner")
        bob = await _research_user("research-intruder")
        thread = f"thread-{uuid4()}"

        job = await research_jobs.create(
            user_id=alice, thread_id=thread, specialist="s", objective="private", concurrency=4
        )
        assert job is not None

        assert await research_jobs.get(job.id, bob) is None
        assert await research_jobs.for_thread(thread, bob) == []
