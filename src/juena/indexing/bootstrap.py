"""Clone the configured repositories, and report which ones are usable.

No indexing happens here any more. juena-rag holds the vectors and an ingestion
job fills them; what this machine still needs is the files themselves, so the
model can open one and quote real line numbers.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juena_core.log import get_logger
from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager

logger = get_logger(__name__)

BOOTSTRAP_DONE_ENV = "JUENA_BOOTSTRAP_DONE"


def clone_repositories(config_path: Path | None = None) -> dict[str, str]:
    """Clone or refresh every configured repository. No indexing.

    Search comes over HTTP, but the model still opens the real file through
    ``/repos/`` to quote real line numbers, and that needs the bytes here.
    """
    repo_manager = RepoManager(load_repo_configs(config_path))
    revisions: dict[str, str] = {}

    for repo_id in repo_manager.repo_ids:
        try:
            repo_manager.resolve_root(repo_id)
            revisions[repo_id] = repo_manager.current_revision(repo_id)
            logger.info("Repository %s at %s", repo_id, revisions[repo_id][:12])
        except Exception:
            logger.warning("Could not clone or refresh %s", repo_id, exc_info=True)

    return revisions


@dataclass(frozen=True)
class RepositorySurvey:
    """Which repositories are usable, and what is wrong with the rest.

    One pass rather than two questions, because asking a remote store twice costs
    a round trip per repository for an answer that cannot have changed in between.
    """

    searchable: list[str]
    problems: list[str]

    @property
    def any_searchable(self) -> bool:
        return bool(self.searchable)


def survey_repositories(
    repo_manager: RepoManager, vector_index: Any
) -> RepositorySurvey:
    """Check every repository once: is it cloned, and does the store hold it?

    A repository needs **both** to be useful -- the store answers the search and
    the clone answers the file read that follows it -- but one broken repository
    is not a reason to take the tools away from the other eight.
    """
    searchable: list[str] = []
    problems: list[str] = []

    if not repo_manager.repo_ids:
        return RepositorySurvey(searchable=[], problems=["no repositories configured"])

    for repo_id in repo_manager.repo_ids:
        cloned = repo_manager.has_local_root(repo_id)
        if not cloned:
            problems.append(f"repository cache missing for '{repo_id}'")

        try:
            chunk_count = vector_index.collection_count_existing(repo_id)
        except Exception as error:
            # An unreachable store is a running problem, not a startup one. Say so
            # and carry on: taking the tools away here would outlive the outage,
            # because the compiled graph is cached for the life of the process.
            problems.append(f"search backend unreachable: {error}")
            return RepositorySurvey(
                searchable=list(repo_manager.repo_ids), problems=problems
            )

        if chunk_count is None:
            problems.append(f"vector index missing for '{repo_id}'")
        elif chunk_count == 0:
            problems.append(f"vector index empty for '{repo_id}'")
        elif cloned:
            searchable.append(repo_id)

    return RepositorySurvey(searchable=searchable, problems=problems)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: clone or refresh every configured repository."""
    argparse.ArgumentParser(prog="juena.indexing.bootstrap").parse_args(argv)

    cloned = clone_repositories()
    logger.info("Cloned or refreshed %d repository(s)", len(cloned))
    os.environ[BOOTSTRAP_DONE_ENV] = "1"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
