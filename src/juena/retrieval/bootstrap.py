"""Repository bootstrap and readiness checks for code-chat startup."""

from __future__ import annotations

import os
from pathlib import Path

from juena.core.log import get_logger
from juena.retrieval.repo_config import load_repo_configs
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex

logger = get_logger(__name__)

BOOTSTRAP_DONE_ENV = "JUENA_BOOTSTRAP_DONE"


def bootstrap_repositories(config_path: Path | None = None) -> dict[str, int]:
    """Clone or refresh configured repositories and rebuild stale indices."""
    configs = load_repo_configs(config_path)
    repo_manager = RepoManager(configs)
    vector_index = RepoVectorIndex(repo_manager)
    results: dict[str, int] = {}
    total_repos = len(repo_manager.repo_ids)

    logger.info("Bootstrapping %d configured repository(s)", total_repos)
    if total_repos == 0:
        logger.info("Bootstrap progress: 100%% (0/0 repositories completed)")
        return results

    logger.info("Bootstrap progress: 0%% (0/%d repositories completed)", total_repos)
    for repo_number, repo_id in enumerate(repo_manager.repo_ids, start=1):
        logger.info(
            "Bootstrapping repository %s (%d/%d)",
            repo_id,
            repo_number,
            total_repos,
        )
        try:
            root = repo_manager.resolve_root(repo_id)
            repo_revision = repo_manager.current_revision(repo_id)
            logger.info("Repository %s synchronized at %s", repo_id, root)
            stale_reason = vector_index.index_staleness_reason(repo_id, repo_revision)
            if stale_reason is None:
                chunk_count = vector_index.collection_count_existing(repo_id) or 0
                logger.info(
                    "Repository %s index is up to date at revision %s (%d chunk(s)); skipping rebuild",
                    repo_id,
                    repo_revision[:12],
                    chunk_count,
                )
            else:
                logger.info(
                    "Rebuilding index for repository %s: %s",
                    repo_id,
                    stale_reason,
                )
                chunk_count = vector_index.build_index(
                    repo_id,
                    force=True,
                    repo_revision=repo_revision,
                )
                logger.info("Repository %s indexed with %d chunk(s)", repo_id, chunk_count)

            results[repo_id] = chunk_count
            progress_percent = int(repo_number * 100 / total_repos)
            logger.info(
                "Bootstrap progress: %d%% (%d/%d repositories completed)",
                progress_percent,
                repo_number,
                total_repos,
            )
        except Exception:
            logger.exception("Bootstrap failed for repository %s", repo_id)
            raise

    return results


def validate_bootstrap_ready(repo_manager: RepoManager, vector_index: RepoVectorIndex) -> None:
    """Ensure repository clones and vector indices already exist locally."""
    problems: list[str] = []

    for repo_id in repo_manager.repo_ids:
        if not repo_manager.has_local_root(repo_id):
            problems.append(f"repository cache missing for '{repo_id}'")
            continue

        chunk_count = vector_index.collection_count_existing(repo_id)
        if chunk_count is None:
            problems.append(f"vector index missing for '{repo_id}'")
        elif chunk_count == 0:
            problems.append(f"vector index empty for '{repo_id}'")

    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "Code-chat bootstrap is incomplete: "
            f"{details}. Run `python -m juena.retrieval.bootstrap` before starting the server."
        )


def main() -> int:
    """CLI entrypoint for repository bootstrap."""
    bootstrap_repositories()
    os.environ[BOOTSTRAP_DONE_ENV] = "1"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
