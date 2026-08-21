"""Every `research_jobs` read and write.

Async SQLAlchemy over the application session factory, not a second psycopg
pool. `sandbox_jobs` needed its own pool because a separate worker process on
the host claims from it; background research runs inside this process, so the
table is ordinary application data like `Chat`.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update

from juena.agents.findings import FindingsMap
from juena.core.log import get_logger
from juena.research.constants import (
    ACTIVE_STATUSES,
    RESTART_MESSAGE,
    ResearchStatus,
)
from juena.server.database.connection import get_session_factory
from juena.server.database.models import ResearchJob, User

logger = get_logger(__name__)


async def create(
    *,
    user_id: UUID,
    thread_id: str,
    specialist: str,
    objective: str,
    concurrency: int,
) -> ResearchJob | None:
    """Admit one job, or return ``None`` when the user is already at the limit.

    The owner's `users` row is locked for the rest of the transaction before the
    count runs. Sharing a transaction is not enough on its own: under
    PostgreSQL's default Read Committed isolation a `COUNT` does not block a
    concurrent `INSERT`, so two calls arriving together could both see the same
    free slot and both admit. The row lock is what serializes them, and it is
    scoped to one user -- two different people starting jobs never wait on each
    other.

    A `users` row that does not exist takes no lock and the insert below then
    fails on the foreign key. That is the right loud failure: `user_id` comes
    from an authenticated context and cannot legitimately be absent.
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
            status="queued",
        )
        session.add(job)
        await session.flush()
        await session.refresh(job)
        return job


async def mark_running(job_id: UUID) -> None:
    factory = get_session_factory()
    async with factory() as session, session.begin():
        await session.execute(
            update(ResearchJob)
            .where(ResearchJob.id == job_id, ResearchJob.status == "queued")
            .values(status="running", started_at=datetime.now(UTC))
        )


async def finish(
    job_id: UUID,
    *,
    status: ResearchStatus,
    report: str = "",
    findings: FindingsMap | None = None,
    base_findings: dict[str, str] | None = None,
    error: str = "",
) -> None:
    """Record a terminal outcome. Only an active job can reach one."""
    factory = get_session_factory()
    async with factory() as session, session.begin():
        await session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status=status,
                report=report,
                findings=findings or {},
                base_findings=base_findings or {},
                error=error,
                finished_at=datetime.now(UTC),
            )
        )


async def get(job_id: UUID, user_id: UUID) -> ResearchJob | None:
    """One job, scoped to its owner. A job id alone is never enough."""
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
    """Jobs on one thread, oldest first.

    Collected jobs are excluded by default: the supervisor asks this question to
    find what it still owes the user, and a result it already reported is not
    that.
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
    it passes grows for the life of the conversation. That makes it **one
    round trip** regardless of length -- not constant work: the `IN` list and the
    index probes behind it are still linear in the number of ids. Round trips are
    what was worth removing; a few hundred uuids in one predicate is not.

    Returns how many rows this actually changed, which is how the caller can log
    a real reconciliation without logging the no-op that every other turn is.
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


async def fail_orphans() -> int:
    """Fail every job left active by a previous process.

    The run itself lived in that process's event loop and did not survive it, so
    a row still reading `running` is a lie the next `check_research` would repeat
    forever. Called once at startup, before anything can create a new job.
    """
    factory = get_session_factory()
    async with factory() as session, session.begin():
        result = await session.execute(
            update(ResearchJob)
            .where(ResearchJob.status.in_(ACTIVE_STATUSES))
            .values(
                status="failed",
                error=RESTART_MESSAGE,
                finished_at=datetime.now(UTC),
            )
        )
    if result.rowcount:
        logger.info("Failed %d research job(s) left by a previous process", result.rowcount)
    return result.rowcount or 0
