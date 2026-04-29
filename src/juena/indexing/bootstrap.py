"""Repository bootstrap and readiness checks for code-chat startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from juena.core.log import get_logger
from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import ManifestDiff, RepoManager, diff_manifests
from juena.indexing.vector_index import RepoVectorIndex

logger = get_logger(__name__)

BOOTSTRAP_DONE_ENV = "JUENA_BOOTSTRAP_DONE"
_MANIFEST_DIR_ENV = "INDEX_MANIFEST_DIR"
_DEFAULT_MANIFEST_DIR = "data/manifests"


def _find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _manifest_dir() -> Path:
    env = os.getenv(_MANIFEST_DIR_ENV)
    if env:
        return Path(env)
    return _find_workspace_root() / _DEFAULT_MANIFEST_DIR


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
    from juena.indexing.chunking import chunk_file

    cfg = repo_manager.get_config(repo_id)
    if cfg is None:
        raise ValueError(f"Unknown repo: {repo_id}")

    files_to_index = diff.added + diff.changed

    for file_path in diff.deleted + diff.changed:
        vector_index.delete_file_chunks(repo_id, file_path)

    for rel_path in files_to_index:
        try:
            content = repo_manager.read_file(repo_id, rel_path)
        except Exception:
            continue

        is_doc = vector_index._is_doc_file(cfg, rel_path)
        chunks = chunk_file(
            content,
            rel_path,
            is_doc=is_doc,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        chunk_dicts: list[dict[str, Any]] = [
            {
                "file_path": c.file_path,
                "chunk_index": c.chunk_index,
                "is_doc": c.is_doc,
                "content_hash": c.content_hash,
                "content": c.text,
            }
            for c in chunks
        ]

        vector_index.upsert_file_chunks(repo_id, chunk_dicts)

    logger.info(
        "Incremental update for %s: %d added, %d changed, %d deleted",
        repo_id,
        len(diff.added),
        len(diff.changed),
        len(diff.deleted),
    )

    return vector_index.collection_count_existing(repo_id) or 0


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
            elif stale_reason == "repository revision changed":
                old_manifest = _load_manifest(repo_id)
                new_manifest = repo_manager.build_manifest(repo_id)
                diff = diff_manifests(old_manifest, new_manifest)

                if old_manifest and diff.has_changes:
                    logger.info(
                        "Incremental reindex for %s: %d added, %d changed, %d deleted",
                        repo_id, len(diff.added), len(diff.changed), len(diff.deleted),
                    )
                    chunk_count = _incremental_reindex(
                        repo_manager, vector_index,
                        repo_id, diff, repo_revision,
                    )
                    _save_manifest(repo_id, new_manifest)
                else:
                    logger.info("Full rebuild for %s (no prior manifest)", repo_id)
                    chunk_count = vector_index.build_index(
                        repo_id, force=True, repo_revision=repo_revision,
                    )
                    _save_manifest(repo_id, new_manifest)
            else:
                logger.info(
                    "Full rebuild for repository %s: %s", repo_id, stale_reason,
                )
                new_manifest = repo_manager.build_manifest(repo_id)
                chunk_count = vector_index.build_index(
                    repo_id, force=True, repo_revision=repo_revision,
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
