"""
RepoManager -- clones and updates the configured git repositories.

It used to walk them too, listing indexable files and hashing them into a
manifest so an index could be updated incrementally. juena-rag now owns
ingestion and re-indexes whole sources, so nothing here reads file contents:
specialists reach the working tree through the read-only ``/repos/`` backend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from juena_core.log import get_logger
from juena.indexing.repo_config import RepoConfig, load_repo_configs
from juena.core.paths import repo_root

logger = get_logger(__name__)

_CACHE_DIR_ENV = "REPO_CACHE_DIR"
_DEFAULT_CACHE_DIR = "data/repos"


def _cache_dir() -> Path:
    env = os.getenv(_CACHE_DIR_ENV)
    if env:
        return Path(env)
    return repo_root() / _DEFAULT_CACHE_DIR


class RepoManager:
    """Manages git-based repository cloning."""

    def __init__(self, configs: list[RepoConfig] | None = None) -> None:
        self._configs: list[RepoConfig] = configs if configs is not None else load_repo_configs()
        self._roots: Dict[str, Path] = {}

    @property
    def repo_ids(self) -> list[str]:
        return [c.id for c in self._configs]

    @property
    def cache_dir(self) -> Path:
        """Return the shared local cache directory for cloned repositories."""
        return _cache_dir()

    def get_config(self, repo_id: str) -> RepoConfig | None:
        for c in self._configs:
            if c.id == repo_id:
                return c
        return None

    def local_root(self, repo_id: str) -> Path:
        """Return the expected local cache path for *repo_id* without syncing it."""
        cfg = self.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")
        return _cache_dir() / cfg.id

    def has_local_root(self, repo_id: str) -> bool:
        """Return whether *repo_id* already exists in the local cache."""
        root = self.local_root(repo_id)
        return root.is_dir() and (root / ".git").is_dir()

    def resolve_root(self, repo_id: str) -> Path:
        """Return the absolute root directory for *repo_id*."""
        if repo_id in self._roots:
            return self._roots[repo_id]

        cfg = self.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        url = cfg.source.url
        if not url:
            raise ValueError(f"Repo {cfg.id}: source.url is required")

        dest = self.local_root(repo_id)
        if (dest / ".git").is_dir():
            self._git_set_origin_url(dest, url)
            self._git_pull(dest, cfg.source.branch)
        elif dest.exists():
            logger.warning(
                "Removing non-git cache path for %s before cloning: %s",
                cfg.id,
                dest,
            )
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
            self._git_clone(url, dest, cfg.source.branch)
        else:
            self._git_clone(url, dest, cfg.source.branch)
        logger.info("Resolved git repo %s -> %s", cfg.id, dest)

        self._roots[repo_id] = dest
        return dest

    def current_revision(self, repo_id: str) -> str:
        """Return the current HEAD revision for *repo_id*."""
        root = self.resolve_root(repo_id)
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "git rev-parse failed for %s (exit %d):\nstdout: %s\nstderr: %s",
                repo_id,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            raise RuntimeError(
                f"git rev-parse failed for {repo_id} (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    @staticmethod
    def _run_git(cmd: list[str], label: str) -> None:
        """Run a git command, logging stderr on failure."""
        logger.info("git: %s – %s", label, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("git %s failed (exit %d):\nstdout: %s\nstderr: %s",
                         label, result.returncode, result.stdout, result.stderr)
            raise RuntimeError(
                f"git {label} failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    @staticmethod
    def _git_clone(url: str, dest: Path, branch: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        RepoManager._run_git(
            ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
            f"clone {url}",
        )

    @staticmethod
    def _git_set_origin_url(dest: Path, url: str) -> None:
        RepoManager._run_git(
            ["git", "-C", str(dest), "remote", "set-url", "origin", url],
            f"set-origin {dest.name}",
        )

    @staticmethod
    def _git_pull(dest: Path, branch: str) -> None:
        RepoManager._run_git(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", branch],
            f"fetch {dest.name}",
        )
        RepoManager._run_git(
            ["git", "-C", str(dest), "reset", "--hard", f"origin/{branch}"],
            f"reset {dest.name}",
        )

