"""Tests for vector index: build + semantic search."""

from pathlib import Path

import pytest

from juena.retrieval.repo_config import load_repo_configs
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex


def _make_index(repo_config_path: Path) -> tuple[RepoManager, RepoVectorIndex]:
    mgr = RepoManager(load_repo_configs(repo_config_path))
    vi = RepoVectorIndex(mgr)
    return mgr, vi


def test_build_and_count(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    count = vi.build_index("fake-repo", force=True)
    assert count > 0
    assert vi.collection_count("fake-repo") == count


def test_semantic_search(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    vi.build_index("fake-repo", force=True)
    hits = vi.search("fake-repo", "greeting function", n_results=3)
    assert len(hits) > 0
    assert all("file_path" in h for h in hits)
    assert all("content" in h for h in hits)


def test_docs_only_filter(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    vi.build_index("fake-repo", force=True)
    hits = vi.search("fake-repo", "guide", n_results=5, where={"is_doc": True})
    assert all(h["is_doc"] for h in hits)
