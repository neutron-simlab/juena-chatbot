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

    def _matches_index_rules(self, cfg: RepoConfig, rel_path: str) -> bool:
        if any(self._matches_glob(rel_path, pat) for pat in cfg.exclude_globs):
            return False
        return any(self._matches_glob(rel_path, pat) for pat in cfg.include_globs)

    def _is_indexable_file(self, cfg: RepoConfig, root: Path, rel_path: str) -> bool:
        if not self._matches_index_rules(cfg, rel_path):
            return False

        full = (root / rel_path).resolve()
        if not str(full).startswith(str(root)):
            return False

        try:
            if not full.is_file():
                return False
            if full.stat().st_size > cfg.max_file_bytes:
                return False
        except OSError:
            return False

        return True

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

                if not self._is_indexable_file(cfg, root, rel):
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

    def diff_revision_files(
        self,
        repo_id: str,
        old_revision: str,
        new_revision: str,
    ) -> "RevisionDiff":
        """Return indexed-file changes between two git revisions."""
        cfg = self.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        root = self.resolve_root(repo_id)
        result = subprocess.run(
            [
                "git", "-C", str(root),
                "diff", "--name-status", "-z", "--find-renames",
                old_revision, new_revision,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"git diff failed for {repo_id} ({old_revision[:12]}..{new_revision[:12]}): {stderr}"
            )

        tokens = result.stdout.split(b"\0")
        if tokens and tokens[-1] == b"":
            tokens.pop()

        added: set[str] = set()
        changed: set[str] = set()
        deleted: set[str] = set()
        renamed: list[tuple[str, str]] = []
        index = 0

        while index < len(tokens):
            status = tokens[index].decode(errors="replace")
            index += 1
            status_code = status[:1]

            if status_code in {"R", "C"}:
                if index + 1 >= len(tokens):
                    break
                old_path = tokens[index].decode(errors="replace")
                new_path = tokens[index + 1].decode(errors="replace")
                index += 2

                old_indexed = self._matches_index_rules(cfg, old_path)
                new_indexed = self._is_indexable_file(cfg, root, new_path)

                if status_code == "C":
                    if new_indexed:
                        added.add(new_path)
                    continue

                if old_indexed and new_indexed:
                    renamed.append((old_path, new_path))
                elif old_indexed:
                    deleted.add(old_path)
                elif new_indexed:
                    added.add(new_path)
                continue

            if index >= len(tokens):
                break
            path = tokens[index].decode(errors="replace")
            index += 1

            if status_code == "D":
                if self._matches_index_rules(cfg, path):
                    deleted.add(path)
                continue

            is_indexable = self._is_indexable_file(cfg, root, path)
            if status_code == "A":
                if is_indexable:
                    added.add(path)
                continue

            if is_indexable:
                changed.add(path)
            elif self._matches_index_rules(cfg, path):
                # A previously indexed file may have grown beyond the size limit.
                deleted.add(path)

        return RevisionDiff(
            added=sorted(added),
            changed=sorted(changed),
            deleted=sorted(deleted),
            renamed=sorted(renamed),
        )

    def update_manifest_for_revision_diff(
        self,
        repo_id: str,
        manifest: dict[str, str],
        diff: "RevisionDiff",
    ) -> dict[str, str]:
        """Apply a revision diff to an existing manifest without rehashing the full repo."""
        updated = dict(manifest)

        for rel_path in diff.deleted:
            updated.pop(rel_path, None)
        for old_path, _new_path in diff.renamed:
            updated.pop(old_path, None)

        for rel_path in diff.added + diff.changed + [new for _, new in diff.renamed]:
            try:
                updated[rel_path] = self.file_hash(repo_id, rel_path)
            except Exception:
                logger.debug("Skipping manifest update for %s/%s", repo_id, rel_path)
                updated.pop(rel_path, None)

        return updated


@dataclass
class ManifestDiff:
    """Result of comparing two file-hash manifests."""

    added: list[str]
    changed: list[str]
    deleted: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.deleted)


@dataclass
class RevisionDiff:
    """Normalized git diff for indexed repository files."""

    added: list[str]
    changed: list[str]
    deleted: list[str]
    renamed: list[tuple[str, str]]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.deleted or self.renamed)

    def as_manifest_diff(self) -> ManifestDiff:
        added = set(self.added)
        deleted = set(self.deleted)

        for old_path, new_path in self.renamed:
            deleted.add(old_path)
            added.add(new_path)

        return ManifestDiff(
            added=sorted(added),
            changed=sorted(self.changed),
            deleted=sorted(deleted),
        )


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
