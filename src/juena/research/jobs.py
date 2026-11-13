"""Research admission, ownership and results in the existing Postgres database."""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sqlalchemy import func, select, update

from juena_core.agents.findings import FindingsMap
from juena_core.config import settings
from juena.research.constants import (
    ACTIVE_STATUSES,
    ResearchStatus,
)
from juena_core.server.database.connection import get_session_factory, psycopg_database_url
from juena_core.server.database.models import User
from juena.server.database.models import ResearchJob


async def create(
    *,
    user_id: UUID,
    thread_id: str,
    specialist: str,
    objective: str,
    concurrency: int,
    run_input: dict[str, Any],
) -> ResearchJob | None:
    """Admit one job, or return ``None`` when the user is already at the limit.

    The owner's `users` row is locked for the rest of the transaction before the
    count runs. Under PostgreSQL's default Read Committed isolation a `COUNT`
    does not block a concurrent `INSERT`, so without the lock two calls arriving
    together could both see the same free slot and both admit. The lock is per
    user, so two different people starting jobs never wait on each other.

    A `users` row that does not exist takes no lock, and the insert then fails
    on the foreign key. That loud failure is right: `user_id` comes from an
    authenticated context and cannot legitimately be absent.

    Args:
        user_id: The job's owner.
        thread_id: The conversation that started the job.
        specialist: Name of the specialist to run.
        objective: The task the specialist receives.
        concurrency: How many active jobs one user may have.
        run_input: Provider, model and files, frozen for the run.

    Returns:
        The new queued job, or None when the user is at the limit.
    """
    factory = get_session_factory()
    async with factory() as session, session.begin():
        await session.execute(select(User.id).where(User.id == user_id).with_for_update())

        active = await session.scalar(
            select(func.count())
            .select_from(ResearchJob)
            .where(
                ResearchJob.user_id == user_id,
                ResearchJob.status.in_(ACTIVE_STATUSES),
            )
        )
        if (active or 0) >= concurrency:
            return None

        job = ResearchJob(
            user_id=user_id,
            thread_id=thread_id,
            specialist=specialist,
            objective=objective,
            run_input=run_input,
            status="queued",
        )
        session.add(job)
        await session.flush()
        await session.refresh(job)
        return job


async def active_ids() -> list[UUID]:
    """List durable work to attempt, including jobs interrupted by a dead process.

    Returns:
        Ids of the queued and running jobs, oldest first.
    """
    factory = get_session_factory()
    async with factory() as session:
        result = await session.scalars(
            select(ResearchJob.id)
            .where(ResearchJob.status.in_(ACTIVE_STATUSES))
            .order_by(ResearchJob.created_at)
        )
        return list(result)


@asynccontextmanager
async def claim(job_id: UUID) -> AsyncIterator[tuple[ResearchJob, AsyncConnection] | None]:
    """Own a job until this connection closes, including on process death.

    The runner uses this same connection for checkpoints and completion. Losing
    ownership therefore also prevents the old runner from persisting progress.
    It is dedicated to the job, so long runs cannot exhaust the chat's pool or
    return an advisory lock to a pool. No lease columns or heartbeat are needed.

    Args:
        job_id: The job to claim.

    Yields:
        The job, now marked running, and its dedicated connection; or None when
        another runner owns it or it is no longer active.
    """
    async with await AsyncConnection.connect(
        psycopg_database_url(settings().DATABASE_URL),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=5,
    ) as connection:
        cursor = await connection.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
            (f"research:{job_id}",),
        )
        if not (await cursor.fetchone())["acquired"]:
            yield None
            return
        cursor = await connection.execute(
            "UPDATE research_jobs SET status = 'running', "
            "started_at = COALESCE(started_at, now()) "
            "WHERE id = %s AND status = ANY(%s) RETURNING *",
            (job_id, list(ACTIVE_STATUSES)),
        )
        row = await cursor.fetchone()
        yield (ResearchJob(**row), connection) if row else None


async def finish(
    connection: AsyncConnection,
    job_id: UUID,
    *,
    status: ResearchStatus,
    report: str = "",
    findings: FindingsMap | None = None,
    base_findings: dict[str, str] | None = None,
    error: str = "",
) -> None:
    """Record the outcome through the connection that owns the job.

    A job that is no longer active is left unchanged.

    Args:
        connection: The connection that claimed the job.
        job_id: The job.
        status: The terminal status.
        report: The specialist's report.
        findings: The files the run wrote.
        base_findings: The content of the run's starting files, by path.
        error: Why the job failed.
    """
    await connection.execute(
        "UPDATE research_jobs SET status = %s, report = %s, findings = %s, "
        "base_findings = %s, error = %s, finished_at = now() "
        "WHERE id = %s AND status = ANY(%s)",
        (status, report, Jsonb(findings or {}), Jsonb(base_findings or {}),
         error, job_id, list(ACTIVE_STATUSES)),
    )


async def get(job_id: UUID, user_id: UUID) -> ResearchJob | None:
    """Return one job, scoped to its owner. A job id alone is never enough.

    Args:
        job_id: The job.
        user_id: The user asking.

    Returns:
        The job, or None when it does not exist or belongs to someone else.
    """
    factory = get_session_factory()
    async with factory() as session:
        return await session.scalar(
            select(ResearchJob).where(
                ResearchJob.id == job_id, ResearchJob.user_id == user_id
            )
        )


async def for_thread(
    thread_id: str, user_id: UUID, *, include_collected: bool = False
) -> list[ResearchJob]:
    """List one user's jobs on one thread, oldest first.

    Collected jobs are excluded by default: the supervisor asks this question to
    find what it still owes the user, and a result it already reported is not
    that.

    Args:
        thread_id: The conversation.
        user_id: The jobs' owner.
        include_collected: Also return jobs already handed over.

    Returns:
        The matching jobs.
    """
    factory = get_session_factory()
    conditions = [ResearchJob.thread_id == thread_id, ResearchJob.user_id == user_id]
    if not include_collected:
        conditions.append(ResearchJob.collected_at.is_(None))
    async with factory() as session:
        result = await session.scalars(
            select(ResearchJob).where(*conditions).order_by(ResearchJob.created_at)
        )
        return list(result)


async def mark_collected(job_id: UUID) -> None:
    """Stamp one job collected, unless it already is.

    Args:
        job_id: The job.
    """
    factory = get_session_factory()
    async with factory() as session, session.begin():
        await session.execute(
            update(ResearchJob)
            .where(ResearchJob.id == job_id, ResearchJob.collected_at.is_(None))
            .values(collected_at=datetime.now(UTC))
        )


async def mark_collected_many(job_ids: Collection[UUID]) -> int:
    """Stamp all of these collected, skipping any already stamped.

    One statement rather than one call per id, because the caller is
    `ResearchDeliveryMiddleware` at the end of every supervisor turn and the list
    it passes grows for the life of the conversation. That makes it one round
    trip regardless of length, though the work is still linear in the number of
    ids; a few hundred ids in one `IN` list is cheap.

    Args:
        job_ids: The jobs to stamp.

    Returns:
        How many rows this actually changed, so the caller can log a real
        reconciliation without logging the no-op that every other turn is.
    """
    if not job_ids:
        return 0
    factory = get_session_factory()
    async with factory() as session, session.begin():
        result = await session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id.in_(list(job_ids)),
                ResearchJob.collected_at.is_(None),
            )
            .values(collected_at=datetime.now(UTC))
        )
    return result.rowcount or 0
