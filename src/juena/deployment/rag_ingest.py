"""Ingest selected repositories, or the application's complete registry."""

import argparse
from pathlib import Path

from juena_rag.ingest.code.pipeline import ingest_repository
from juena_rag.ingest.code.repository import load_repositories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repositories", nargs="*")
    parser.add_argument("--config", type=Path, default=Path("/app/config/repositories.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/rag-repos"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    configured = load_repositories(args.config)
    ids = args.repositories or list(configured)
    if unknown := set(ids) - configured.keys():
        parser.error(f"Unknown repositories: {', '.join(sorted(unknown))}")
    failed = []
    for repo_id in ids:
        result = ingest_repository(
            repo_id, config_path=args.config, cache_dir=args.cache_dir,
            dry_run=args.dry_run, no_sync=args.no_sync, force=args.force, limit=args.limit,
        )
        if result:
            failed.append(repo_id)
    if failed:
        print(f"Ingestion failed for: {', '.join(failed)}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
