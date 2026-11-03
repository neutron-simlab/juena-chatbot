"""Read-only production capability checks. Never print credentials or raw errors.

The launcher pipes this file into the running image, so diagnostics also work
before deploying a new image.
"""

import json


def repository_report(manager, index) -> dict:
    report = {"configured": list(manager.repo_ids)}
    try:
        index.ping()
        report["reachable"] = True
        report["indexed"] = index.catalogue()
        report["cloned"] = [repo for repo in manager.repo_ids if manager.has_local_root(repo)]
        report["needs_ingestion"] = not any(row["chunks"] > 0 for row in report["indexed"])
    except Exception as exc:
        # HTTP/database exception messages may contain URLs with credentials.
        report["error_type"] = type(exc).__name__
    return report


def sandbox_report(config) -> dict:
    import psycopg
    from juena_core.sandbox.jobs import (
        WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID, database_url,
    )

    report = {"enabled": config.SANDBOX_ENABLED}
    try:
        with psycopg.connect(database_url(config.DATABASE_URL), connect_timeout=5) as connection:
            report["jobs_table_exists"] = connection.execute(
                "SELECT to_regclass('public.sandbox_jobs') IS NOT NULL"
            ).fetchone()[0]
            report["worker_connected"] = bool(connection.execute(
                "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND classid = %s "
                "AND objid = %s AND objsubid = 2 AND granted LIMIT 1",
                (WORKER_LOCK_CLASSID, WORKER_LOCK_OBJID),
            ).fetchone())
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    return report


def main() -> int:
    from juena.core.config import global_config
    from juena.indexing.rag_index import RagVectorIndex
    from juena.indexing.repo_manager import RepoManager

    report = {
        "rag": repository_report(RepoManager(), RagVectorIndex(global_config.JUENA_RAG_URL, timeout=10)),
        "sandbox": sandbox_report(global_config),
    }
    print(json.dumps(report, indent=2))
    return int(bool(report["rag"].get("error_type") or report["sandbox"].get("error_type")))


if __name__ == "__main__":
    raise SystemExit(main())
