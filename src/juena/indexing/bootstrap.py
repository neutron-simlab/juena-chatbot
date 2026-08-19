"""Repository bootstrap and readiness checks for code-chat startup."""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juena.core.log import get_logger
from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import (
    ManifestDiff,
    RepoManager,
    RevisionDiff,
    diff_manifests,
)
from juena.indexing.vector_index import RepoVectorIndex
from juena.core.paths import repo_root

logger = get_logger(__name__)

BOOTSTRAP_DONE_ENV = "JUENA_BOOTSTRAP_DONE"
_MANIFEST_DIR_ENV = "INDEX_MANIFEST_DIR"
_DEFAULT_MANIFEST_DIR = "data/manifests"


def _manifest_dir() -> Path:
    env = os.getenv(_MANIFEST_DIR_ENV)
    if env:
        return Path(env)
    return repo_root() / _DEFAULT_MANIFEST_DIR


def _load_manifest(repo_id: str) -> dict[str, str]:
    path = _manifest_dir() / f"{repo_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(repo_id: str, manifest: dict[str, str]) -> None:
    d = _manifest_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{repo_id}.json", "w") as f:
        json.dump(manifest, f, sort_keys=True)


def _incremental_reindex(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    diff: ManifestDiff,
    repo_revision: str,
) -> int:
    """Re-index only files that changed. Returns total chunk count after update."""
    files_to_index = diff.added + diff.changed

    for file_path in diff.deleted + diff.changed:
        vector_index.delete_file_chunks(repo_id, file_path)

    for rel_path in files_to_index:
        try:
            chunk_dicts = vector_index.build_file_chunks(repo_id, rel_path)
        except Exception:
            continue

        vector_index.upsert_file_chunks(repo_id, chunk_dicts)

    logger.info(
        "Incremental update for %s: %d added, %d changed, %d deleted",
        repo_id,
        len(diff.added),
        len(diff.changed),
        len(diff.deleted),
    )

    return vector_index.collection_count_existing(repo_id) or 0


def _bootstrap_from_revision_diff(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    repo_revision: str,
    revision_diff: RevisionDiff,
    old_manifest: dict[str, str],
) -> tuple[int, dict[str, str] | None]:
    """Apply a git revision diff and update metadata without rebuilding the full index."""
    manifest_diff = revision_diff.as_manifest_diff()

    if manifest_diff.has_changes:
        logger.info(
            "Incremental reindex for %s via git diff: %d added, %d changed, %d deleted, %d renamed",
            repo_id,
            len(revision_diff.added),
            len(revision_diff.changed),
            len(revision_diff.deleted),
            len(revision_diff.renamed),
        )
        chunk_count = _incremental_reindex(
            repo_manager,
            vector_index,
            repo_id,
            manifest_diff,
            repo_revision,
        )
    else:
        chunk_count = vector_index.collection_count_existing(repo_id) or 0
        logger.info(
            "Repository %s revision changed but indexed files did not; updating metadata only",
            repo_id,
        )

    vector_index.update_index_metadata(repo_id, repo_revision)

    if old_manifest:
        return (
            chunk_count,
            repo_manager.update_manifest_for_revision_diff(
                repo_id, old_manifest, revision_diff
            ),
        )

    return chunk_count, repo_manager.build_manifest(repo_id)


def _bootstrap_from_manifest_diff(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    repo_revision: str,
    old_manifest: dict[str, str],
) -> tuple[int, dict[str, str]]:
    """Fallback incremental path when git diff cannot be trusted."""
    new_manifest = repo_manager.build_manifest(repo_id)
    diff = diff_manifests(old_manifest, new_manifest)

    if diff.has_changes:
        logger.info(
            "Incremental reindex for %s via manifest diff: %d added, %d changed, %d deleted",
            repo_id,
            len(diff.added),
            len(diff.changed),
            len(diff.deleted),
        )
        chunk_count = _incremental_reindex(
            repo_manager,
            vector_index,
            repo_id,
            diff,
            repo_revision,
        )
    else:
        chunk_count = vector_index.collection_count_existing(repo_id) or 0
        logger.info(
            "Repository %s manifest matches existing index; updating metadata only",
            repo_id,
        )

    vector_index.update_index_metadata(repo_id, repo_revision)
    return chunk_count, new_manifest


def bootstrap_repositories(config_path: Path | None = None) -> dict[str, int]:
    """Clone or refresh configured repositories and rebuild stale indices.

    Retired with the Chroma cutover and kept only so the benchmark can still
    build a Chroma arm -- nothing in the running application calls it. Startup
    clones, and juena-rag holds the vectors.
    """
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
            elif stale_reason == "repository revision changed":
                old_manifest = _load_manifest(repo_id)
                indexed_revision = vector_index.indexed_revision(repo_id)

                if indexed_revision:
                    try:
                        revision_diff = repo_manager.diff_revision_files(
                            repo_id,
                            indexed_revision,
                            repo_revision,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Git diff unavailable for %s (%s..%s): %s",
                            repo_id,
                            indexed_revision[:12],
                            repo_revision[:12],
                            exc,
                        )
                    else:
                        chunk_count, new_manifest = _bootstrap_from_revision_diff(
                            repo_manager,
                            vector_index,
                            repo_id,
                            repo_revision,
                            revision_diff,
                            old_manifest,
                        )
                        _save_manifest(repo_id, new_manifest)
                        results[repo_id] = chunk_count
                        progress_percent = int(repo_number * 100 / total_repos)
                        logger.info(
                            "Bootstrap progress: %d%% (%d/%d repositories completed)",
                            progress_percent,
                            repo_number,
                            total_repos,
                        )
                        continue

                if old_manifest:
                    chunk_count, new_manifest = _bootstrap_from_manifest_diff(
                        repo_manager,
                        vector_index,
                        repo_id,
                        repo_revision,
                        old_manifest,
                    )
                    _save_manifest(repo_id, new_manifest)
                else:
                    logger.info(
                        "Full rebuild for %s (no prior manifest baseline)", repo_id
                    )
                    new_manifest = repo_manager.build_manifest(repo_id)
                    chunk_count = vector_index.build_index(
                        repo_id,
                        force=True,
                        repo_revision=repo_revision,
                    )
                    _save_manifest(repo_id, new_manifest)
            else:
                logger.info(
                    "Full rebuild for repository %s: %s",
                    repo_id,
                    stale_reason,
                )
                new_manifest = repo_manager.build_manifest(repo_id)
                chunk_count = vector_index.build_index(
                    repo_id,
                    force=True,
                    repo_revision=repo_revision,
                )
                _save_manifest(repo_id, new_manifest)

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


def clone_repositories(config_path: Path | None = None) -> dict[str, str]:
    """Clone or refresh every configured repository. No indexing.

    The half of :func:`bootstrap_repositories` this pod still needs once the store
    is remote: search comes over HTTP, but the model still opens the real file
    through ``/repos/`` to quote real line numbers, and that needs the bytes here.
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
    is not a reason to take the tools away from the other eight, which is what
    `describe_bootstrap_problems` used to cause.
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


def describe_bootstrap_problems(
    repo_manager: RepoManager, vector_index: RepoVectorIndex
) -> list[str]:
    """Return the reasons repository research is unavailable, empty when ready.

    Callers that can degrade gracefully should use this instead of
    :func:`validate_bootstrap_ready`, which raises.
    """
    problems: list[str] = []

    if not repo_manager.repo_ids:
        return ["no repositories configured"]

    for repo_id in repo_manager.repo_ids:
        if not repo_manager.has_local_root(repo_id):
            problems.append(f"repository cache missing for '{repo_id}'")
            continue

        chunk_count = vector_index.collection_count_existing(repo_id)
        if chunk_count is None:
            problems.append(f"vector index missing for '{repo_id}'")
        elif chunk_count == 0:
            problems.append(f"vector index empty for '{repo_id}'")

    return problems


def validate_bootstrap_ready(
    repo_manager: RepoManager, vector_index: RepoVectorIndex
) -> None:
    """Ensure repository clones and vector indices already exist locally."""
    problems = describe_bootstrap_problems(repo_manager, vector_index)
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "Code-chat bootstrap is incomplete: "
            f"{details}. Run `python -m juena.indexing.bootstrap` before starting the server."
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: clone or refresh every configured repository.

    It no longer builds an index. juena-rag holds the vectors and an ingestion
    job fills them; what this machine still needs is the files, so the model can
    open one and quote real line numbers.
    """
    argparse.ArgumentParser(prog="juena.indexing.bootstrap").parse_args(argv)

    cloned = clone_repositories()
    logger.info("Cloned or refreshed %d repository(s)", len(cloned))
    os.environ[BOOTSTRAP_DONE_ENV] = "1"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
