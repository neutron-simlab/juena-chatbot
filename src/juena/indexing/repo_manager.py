"""
RepoManager – clones/updates git repos and builds a per-repo file list
honouring include/exclude globs and size limits.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from juena.core.log import get_logger
from juena.indexing.repo_config import RepoConfig, load_repo_configs

logger = get_logger(__name__)

_CACHE_DIR_ENV = "REPO_CACHE_DIR"
_DEFAULT_CACHE_DIR = "data/repos"


def _find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _cache_dir() -> Path:
    env = os.getenv(_CACHE_DIR_ENV)
    if env:
        return Path(env)
    return _find_workspace_root() / _DEFAULT_CACHE_DIR


class RepoManager:
    """Manages git-based repository cloning and file listing."""

    def __init__(self, configs: list[RepoConfig] | None = None) -> None:
        self._configs: list[RepoConfig] = configs if configs is not None else load_repo_configs()
        self._roots: Dict[str, Path] = {}
        self._file_cache: Dict[str, List[str]] = {}

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

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_glob(rel_path: str, pattern: str) -> bool:
        """Match *rel_path* against *pattern*, handling ``**/`` for zero+ dirs."""
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        if pattern.startswith("**/"):
            return RepoManager._matches_glob(rel_path, pattern[3:])
        return False

    def list_files(self, repo_id: str, *, force: bool = False) -> list[str]:
        """Return repo-relative file paths that match include/exclude rules."""
        if not force and repo_id in self._file_cache:
            return self._file_cache[repo_id]

        cfg = self.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        root = self.resolve_root(repo_id)
        matched: list[str] = []

        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                abs_path = Path(dirpath) / fname
                rel = str(abs_path.relative_to(root))

                if any(self._matches_glob(rel, pat) for pat in cfg.exclude_globs):
                    continue
                if not any(self._matches_glob(rel, pat) for pat in cfg.include_globs):
                    continue
                try:
                    if abs_path.stat().st_size > cfg.max_file_bytes:
                        continue
                except OSError:
                    continue

                matched.append(rel)

        matched.sort()
        self._file_cache[repo_id] = matched
        logger.info("Repo %s: %d files match include/exclude rules", repo_id, len(matched))
        return matched

    def read_file(self, repo_id: str, rel_path: str) -> str:
        """Read file content, raising FileNotFoundError if missing."""
        root = self.resolve_root(repo_id)
        full = (root / rel_path).resolve()
        if not str(full).startswith(str(root)):
            raise ValueError("Path traversal detected")
        return full.read_text(errors="replace")

    def list_repo_metadata(self) -> list[dict]:
        """Return lightweight metadata dicts for all configured repos."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "source_url": c.source.url,
            }
            for c in self._configs
        ]

    # ------------------------------------------------------------------
    # File-hash manifest for incremental indexing
    # ------------------------------------------------------------------

    def file_hash(self, repo_id: str, rel_path: str) -> str:
        """Return a SHA-256 content hash for a single file."""
        root = self.resolve_root(repo_id)
        full = (root / rel_path).resolve()
        h = hashlib.sha256()
        h.update(full.read_bytes())
        return h.hexdigest()[:16]

    def build_manifest(self, repo_id: str) -> dict[str, str]:
        """Return ``{rel_path: content_hash}`` for all indexed files."""
        files = self.list_files(repo_id)
        manifest: dict[str, str] = {}
        for rel_path in files:
            try:
                manifest[rel_path] = self.file_hash(repo_id, rel_path)
            except Exception:
                logger.debug("Skipping manifest entry for %s/%s", repo_id, rel_path)
        return manifest


@dataclass
class ManifestDiff:
    """Result of comparing two file-hash manifests."""

    added: list[str]
    changed: list[str]
    deleted: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.deleted)


def diff_manifests(
    old: dict[str, str],
    new: dict[str, str],
) -> ManifestDiff:
    """Compare two ``{path: hash}`` manifests and return the diff."""
    old_keys = set(old)
    new_keys = set(new)
    added = sorted(new_keys - old_keys)
    deleted = sorted(old_keys - new_keys)
    changed = sorted(
        p for p in old_keys & new_keys
        if old[p] != new[p]
    )
    return ManifestDiff(added=added, changed=changed, deleted=deleted)
