"""Run Postgres sandbox jobs in fixed rootless-Podman slots."""

from __future__ import annotations

import logging
import os
import signal
import threading
from datetime import UTC, datetime, timedelta

from juena.sandbox.executor import ContainerCleanupError, PodmanExecutor, SandboxSettings
from juena.sandbox.jobs import (
    CLAIM_DEADLINE_SECONDS,
    CLAIM_POLL_SECONDS,
    SandboxJobRecord,
    SandboxJobs,
    hold_worker_lock,
)
from juena.sandbox.workspace import WorkspaceStore

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 3600
TERMINAL_RETENTION_HOURS = 24
RESTART_MESSAGE = "Sandbox worker restarted before this job completed"


class SandboxWorker:
    def __init__(self, settings: SandboxSettings) -> None:
        self.settings = settings
        self.jobs = SandboxJobs(settings.database_url, concurrency=settings.concurrency)
        self.workspaces = WorkspaceStore(
            settings.workspace_root,
            limit_bytes=settings.workspace_limit_bytes,
            idle_ttl_seconds=settings.idle_ttl_seconds,
        )
        self.executor = PodmanExecutor(settings, self.workspaces)

    def _execute(self, job: SandboxJobRecord) -> None:
        try:
            outcome = self.executor.run(job)
        except Exception as exc:
            logger.exception("Sandbox job %s failed in the worker", job.id)
            self.jobs.finish(job.id, status="failed", output=f"Sandbox job failed: {exc}")
            if isinstance(exc, ContainerCleanupError):
                raise
            return
        self.jobs.finish(
            job.id,
            status=outcome.status,
            output=outcome.output,
            exit_code=outcome.exit_code,
            truncated=outcome.truncated,
            artifact_names=outcome.artifact_names,
        )

    def _slot(self, stop: threading.Event, fatal: list[BaseException]) -> None:
        try:
            while not stop.is_set():
                job = self.jobs.claim()
                if job is None:
                    stop.wait(CLAIM_POLL_SECONDS)
                    continue
                self._execute(job)
        except BaseException as exc:
            logger.exception("Sandbox worker slot failed")
            fatal.append(exc)
            stop.set()

    def _cleanup(self, stop: threading.Event) -> None:
        cycles = 0
        while not stop.wait(CLAIM_DEADLINE_SECONDS):
            try:
                stale = self.jobs.fail_stale_queued()
                cycles += 1
                if cycles < CLEANUP_INTERVAL_SECONDS // CLAIM_DEADLINE_SECONDS:
                    if stale:
                        logger.info("Failed %d stale queued sandbox jobs", stale)
                    continue
                cycles = 0
                removed = self.jobs.cleanup(
                    older_than=datetime.now(UTC) - timedelta(hours=TERMINAL_RETENTION_HOURS)
                )
                expired = self.workspaces.cleanup_expired(self.jobs.active_workspaces())
                logger.info(
                    "Sandbox cleanup: %d stale, %d rows removed, %d workspaces removed",
                    stale,
                    removed,
                    expired,
                )
            except Exception:
                logger.exception("Sandbox cleanup pass failed")

    def run(self, stop: threading.Event) -> None:
        self.jobs.open()
        self.executor.connect()
        self.executor.validate_startup()
        logger.info("Removed %d leftover containers", self.executor.remove_all())
        logger.info("Failed %d interrupted jobs", self.jobs.fail_active(RESTART_MESSAGE))

        fatal: list[BaseException] = []
        threads = [
            threading.Thread(
                target=self._slot,
                args=(stop, fatal),
                name=f"sandbox-slot-{index}",
            )
            for index in range(self.settings.concurrency)
        ]
        threads.append(
            threading.Thread(target=self._cleanup, args=(stop,), name="sandbox-cleanup")
        )
        for thread in threads:
            thread.daemon = True
            thread.start()
        logger.info("Sandbox worker ready with %d slots", self.settings.concurrency)
        try:
            stop.wait()
        finally:
            for thread in threads:
                thread.join(timeout=30)
            self.executor.close()
            self.jobs.close()
        if fatal:
            raise RuntimeError("Sandbox worker stopped after a slot failure") from fatal[0]


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    settings = SandboxSettings.from_env()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with hold_worker_lock(settings.database_url) as acquired:
        if not acquired:
            logger.error("Another sandbox worker already holds the worker lock")
            return 1
        SandboxWorker(settings).run(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
