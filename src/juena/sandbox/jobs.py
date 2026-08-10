"""Postgres-backed sandbox admission, claiming, and results."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from juena.sandbox.constants import ACTIVE_STATUSES, TERMINAL_STATUSES, JobStatus

# The worker holds this session lock for its lifetime; JüNA only observes it.
WORKER_LOCK_CLASSID = 0x4A55
WORKER_LOCK_OBJID = 0x454E

# Queued rows are short-lived claims, not a waiting queue.
CLAIM_DEADLINE_SECONDS = 15
CLAIM_POLL_SECONDS = 0.25
RESULT_POLL_SECONDS = 0.25

_ADMISSION_LOCK = 0x4A55454E41534E44
_DATABASE_URL_PREFIXES = ("postgresql+psycopg://", "postgresql://", "postgres://")
_COLUMNS = (
    "id, tenant_id, workspace_id, command, timeout_seconds, max_output_bytes, "
    "status, output, exit_code, truncated, artifact_names, created_at, "
    "started_at, finished_at"
)


def database_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix psycopg does not understand."""
    for prefix in _DATABASE_URL_PREFIXES:
        if url.startswith(prefix):
            return f"postgresql://{url.removeprefix(prefix)}"
    raise ValueError("DATABASE_URL must use postgres:// or postgresql://")


def tenant_id_for(user_id: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()


def workspace_id_for(user_id: str, thread_id: str, secret: str) -> str:
    payload = f"{user_id}\0{thread_id}".encode()
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _workspace_lock_key(workspace_id: str) -> int:
    digest = hashlib.sha256(f"workspace:{workspace_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@dataclass(frozen=True, slots=True)
class SandboxJobRecord:
    id: UUID
    tenant_id: str
    workspace_id: str
    command: str
    timeout_seconds: int
    max_output_bytes: int
    status: JobStatus
    output: str
    exit_code: int | None
    truncated: bool
    artifact_names: tuple[str, ...]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    status: str
    output: str
    job_id: UUID | None = None
    exit_code: int | None = None
    truncated: bool = False
    artifact_names: tuple[str, ...] = ()


def _record(row: dict) -> SandboxJobRecord:
    return SandboxJobRecord(
        id=row["id"],
        tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"],
        command=row["command"],
        timeout_seconds=row["timeout_seconds"],
        max_output_bytes=row["max_output_bytes"],
        status=row["status"],
        output=row["output"],
        exit_code=row["exit_code"],
        truncated=row["truncated"],
        artifact_names=tuple(row["artifact_names"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _result(record: SandboxJobRecord) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        status=record.status,
        output=record.output,
        job_id=record.id,
        exit_code=record.exit_code,
        truncated=record.truncated,
        artifact_names=record.artifact_names,
    )


@contextmanager
def hold_worker_lock(url: str) -> Iterator[bool]:
    """Hold the worker session lock on a dedicated connection."""

    with psycopg.connect(database_url(url), autocommit=True) as connection:
        row = connection.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID),
        ).fetchone()
        yield bool(row and row[0])


class SandboxJobs:
    """Every `sandbox_jobs` operation, for both the application and the worker."""

    def __init__(self, url: str, *, concurrency: int) -> None:
        self.concurrency = concurrency
        self.pool = ConnectionPool(
            conninfo=database_url(url),
            min_size=0,
            max_size=max(concurrency + 2, 4),
            open=False,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        self.pool.open(wait=True)

    def close(self) -> None:
        self.pool.close()

    def worker_is_alive(self) -> bool:
        """Read the worker's session lock. objsubid 2 marks the two-int4 form."""

        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND classid = %s "
                "AND objid = %s AND objsubid = 2 AND granted LIMIT 1",
                (WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID),
            ).fetchone()
        return row is not None

    def create(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        command: str,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> SandboxJobRecord | None:
        """Admit a job without racing capacity or workspace file operations."""

        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_ADMISSION_LOCK,))
            workspace = connection.execute(
                "SELECT pg_try_advisory_xact_lock(%s) AS locked",
                (_workspace_lock_key(workspace_id),),
            ).fetchone()
            if not workspace or not workspace["locked"]:
                return None
            active = connection.execute(
                "SELECT count(*) AS count FROM sandbox_jobs WHERE status = ANY(%s)",
                (list(ACTIVE_STATUSES),),
            ).fetchone()
            if active is None or active["count"] >= self.concurrency:
                return None
            row = connection.execute(
                f"""
                INSERT INTO sandbox_jobs (
                    id, tenant_id, workspace_id, command, timeout_seconds,
                    max_output_bytes, status, output, truncated, artifact_names,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'queued', '', false, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING {_COLUMNS}
                """,
                (
                    uuid4(),
                    tenant_id,
                    workspace_id,
                    command,
                    timeout_seconds,
                    max_output_bytes,
                    Jsonb([]),
                    datetime.now(UTC),
                ),
            ).fetchone()
        return _record(row) if row is not None else None

    def execute(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        command: str,
        timeout_seconds: int,
        max_output_bytes: int,
        status_callback: Callable[[str], None] | None = None,
    ) -> SandboxExecutionResult:
        """Submit one command and wait for its terminal row."""

        if not self.worker_is_alive():
            return SandboxExecutionResult("unavailable", "Sandbox worker is unavailable")
        job = self.create(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            command=command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if job is None:
            return SandboxExecutionResult("busy", "Sandbox capacity or workspace is busy")
        if status_callback:
            status_callback("queued")

        deadline = monotonic() + CLAIM_DEADLINE_SECONDS + timeout_seconds + 30
        claim_deadline = job.created_at + timedelta(seconds=CLAIM_DEADLINE_SECONDS)
        last_status = "queued"
        while monotonic() < deadline:
            sleep(RESULT_POLL_SECONDS)
            record = self.get(job.id)
            if record is None:
                return SandboxExecutionResult("failed", "Sandbox job disappeared", job.id)
            if record.status in TERMINAL_STATUSES:
                return _result(record)
            if record.status == "queued" and datetime.now(UTC) > claim_deadline:
                failed = self.fail_queued(job.id, "Sandbox worker is unavailable")
                if failed:
                    return _result(failed)
                continue
            if record.status != last_status:
                last_status = record.status
                if status_callback:
                    status_callback(record.status)
        return _result(
            self.finish(job.id, status="failed", output="Sandbox result wait timed out")
            or job
        )

    @contextmanager
    def workspace_guard(self, workspace_id: str) -> Iterator[bool]:
        """Yield False instead of blocking behind an active workspace job."""

        with self.pool.connection() as connection, connection.transaction():
            locked = connection.execute(
                "SELECT pg_try_advisory_xact_lock(%s) AS locked",
                (_workspace_lock_key(workspace_id),),
            ).fetchone()
            active = connection.execute(
                "SELECT 1 FROM sandbox_jobs WHERE workspace_id = %s "
                "AND status = ANY(%s) LIMIT 1",
                (workspace_id, list(ACTIVE_STATUSES)),
            ).fetchone()
            yield bool(locked and locked["locked"] and active is None)

    def claim(self) -> SandboxJobRecord | None:
        """Take the oldest queued job. SKIP LOCKED keeps slots from colliding."""

        now = datetime.now(UTC)
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs
                SET status = 'running', started_at = %s
                WHERE id = (
                    SELECT id FROM sandbox_jobs
                    WHERE status = 'queued' AND created_at >= %s
                    ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                RETURNING {_COLUMNS}
                """,
                (now, now - timedelta(seconds=CLAIM_DEADLINE_SECONDS)),
            ).fetchone()
        return _record(row) if row is not None else None

    def get(self, job_id: UUID) -> SandboxJobRecord | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM sandbox_jobs WHERE id = %s",
                (job_id,),
            ).fetchone()
        return _record(row) if row is not None else None

    def finish(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        output: str,
        exit_code: int | None = None,
        truncated: bool = False,
        artifact_names: list[str] | None = None,
    ) -> SandboxJobRecord | None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Sandbox job status is not terminal: {status}")
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs
                SET status = %s, output = %s, exit_code = %s, truncated = %s,
                    artifact_names = %s, finished_at = %s
                WHERE id = %s AND status = ANY(%s)
                RETURNING {_COLUMNS}
                """,
                (
                    status,
                    output,
                    exit_code,
                    truncated,
                    Jsonb(artifact_names or []),
                    datetime.now(UTC),
                    job_id,
                    list(ACTIVE_STATUSES),
                ),
            ).fetchone()
        return _record(row) if row is not None else self.get(job_id)

    def fail_active(self, output: str) -> int:
        """Fail every row left by a previous worker process."""

        with self.pool.connection() as connection:
            return connection.execute(
                "UPDATE sandbox_jobs SET status = 'failed', output = %s, "
                "finished_at = %s WHERE status = ANY(%s)",
                (output, datetime.now(UTC), list(ACTIVE_STATUSES)),
            ).rowcount

    def fail_stale_queued(self) -> int:
        now = datetime.now(UTC)
        with self.pool.connection() as connection:
            return connection.execute(
                "UPDATE sandbox_jobs SET status = 'failed', "
                "output = 'Sandbox worker was unavailable', finished_at = %s "
                "WHERE status = 'queued' AND created_at < %s",
                (now, now - timedelta(seconds=CLAIM_DEADLINE_SECONDS)),
            ).rowcount

    def fail_queued(self, job_id: UUID, output: str) -> SandboxJobRecord | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                f"""
                UPDATE sandbox_jobs SET status = 'failed', output = %s, finished_at = %s
                WHERE id = %s AND status = 'queued'
                RETURNING {_COLUMNS}
                """,
                (output, datetime.now(UTC), job_id),
            ).fetchone()
        return _record(row) if row is not None else None

    def active_workspaces(self) -> set[str]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT workspace_id FROM sandbox_jobs WHERE status = ANY(%s)",
                (list(ACTIVE_STATUSES),),
            ).fetchall()
        return {row["workspace_id"] for row in rows}

    def cleanup(self, *, older_than: datetime) -> int:
        with self.pool.connection() as connection:
            return connection.execute(
                "DELETE FROM sandbox_jobs WHERE status = ANY(%s) AND finished_at < %s",
                (list(TERMINAL_STATUSES), older_than),
            ).rowcount
