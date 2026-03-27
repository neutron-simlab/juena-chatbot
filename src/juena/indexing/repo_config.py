"""
Repository registry configuration – loads ``config/repositories.yaml``
and exposes typed dataclass objects for the rest of the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from juena.core.log import get_logger

logger = get_logger(__name__)

_DEFAULT_INCLUDE = [
    "**/*.py", "**/*.md", "**/*.json", "**/*.yaml", "**/*.yml",
    "**/*.toml", "**/*.txt", "**/*.rst",
]
_DEFAULT_EXCLUDE = [
    "**/node_modules/**", "**/.git/**", "**/__pycache__/**",
    "**/.venv/**", "**/dist/**", "**/build/**", "**/*.egg-info/**",
]


@dataclass
class RepoSource:
    url: str
    branch: str = "main"


@dataclass
class RepoConfig:
    id: str
    name: str
    description: str = ""
    source: RepoSource = field(default_factory=lambda: RepoSource(url=""))
    include_globs: List[str] = field(default_factory=lambda: list(_DEFAULT_INCLUDE))
    exclude_globs: List[str] = field(default_factory=lambda: list(_DEFAULT_EXCLUDE))
    max_file_bytes: int = 524_288  # 512 KB
    chunk_size: int = 1500
    chunk_overlap: int = 200
    docs_paths: List[str] = field(default_factory=lambda: ["README.md", "docs/"])


def _find_workspace_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _resolve_config_path() -> Path:
    """Return the absolute path to ``config/repositories.yaml``."""
    env_path = os.getenv("REPO_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _find_workspace_root() / "config" / "repositories.yaml"


def load_repo_configs(config_path: Path | None = None) -> list[RepoConfig]:
    """Parse the YAML config and return a list of ``RepoConfig`` objects."""
    config_path = config_path or _resolve_config_path()
    if not config_path.exists():
        logger.warning("Repository config not found at %s – no repos will be indexed", config_path)
        return []

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    repos: list[RepoConfig] = []
    for entry in raw.get("repositories", []):
        src_raw = entry.get("source", {})
        url = src_raw.get("url", "")
        if not url:
            logger.warning("Repo %s: missing source.url – skipping", entry.get("id", "?"))
            continue
        source = RepoSource(
            url=url,
            branch=src_raw.get("branch", "main"),
        )
        repos.append(
            RepoConfig(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                description=entry.get("description", ""),
                source=source,
                include_globs=entry.get("include_globs", list(_DEFAULT_INCLUDE)),
                exclude_globs=entry.get("exclude_globs", list(_DEFAULT_EXCLUDE)),
                max_file_bytes=entry.get("max_file_bytes", 524_288),
                chunk_size=entry.get("chunk_size", 1500),
                chunk_overlap=entry.get("chunk_overlap", 200),
                docs_paths=entry.get("docs_paths", ["README.md", "docs/"]),
            )
        )
    logger.info("Loaded %d repository config(s): %s", len(repos), [r.id for r in repos])
    return repos
