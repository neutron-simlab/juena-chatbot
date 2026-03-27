"""Tests for sparse (FTS5) index: build + lexical search."""

import os
from pathlib import Path

import pytest

from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager
from juena.indexing.sparse_index import RepoSparseIndex, SparseHit


@pytest.fixture(autouse=True)
def _sparse_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARSE_INDEX_DIR", str(tmp_path / "sparse"))


def _make_index(repo_config_path: Path) -> tuple[RepoManager, RepoSparseIndex]:
    mgr = RepoManager(load_repo_configs(repo_config_path))
    si = RepoSparseIndex(mgr)
    return mgr, si


def _sample_chunks() -> list[dict]:
    return [
        {
            "file_path": "src/main.py",
            "chunk_index": 0,
            "is_doc": False,
            "content_hash": "abc123",
            "content": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"',
        },
        {
            "file_path": "README.md",
            "chunk_index": 0,
            "is_doc": True,
            "content_hash": "def456",
            "content": "# Fake Repo\n\nA tiny test repository.",
        },
        {
            "file_path": "src/utils.py",
            "chunk_index": 0,
            "is_doc": False,
            "content_hash": "ghi789",
            "content": "import os\n\ndef get_env(key: str, default: str = \"\") -> str:\n    return os.getenv(key, default)",
        },
    ]


def test_build_and_count(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    count = si.build_index("fake-repo", _sample_chunks(), force=True)
    assert count == 3
    assert si.chunk_count("fake-repo") == 3


def test_search_finds_keyword(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    si.build_index("fake-repo", _sample_chunks(), force=True)

    hits = si.search("fake-repo", "greet", max_results=5)
    assert len(hits) > 0
    assert isinstance(hits[0], SparseHit)
    assert any("main.py" in h.file_path for h in hits)


def test_search_docs_only(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    si.build_index("fake-repo", _sample_chunks(), force=True)

    hits = si.search("fake-repo", "repository", max_results=5, docs_only=True)
    assert all(h.is_doc for h in hits)


def test_search_no_match(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    si.build_index("fake-repo", _sample_chunks(), force=True)

    hits = si.search("fake-repo", "zzz_nonexistent_symbol_zzz", max_results=5)
    assert hits == []


def test_delete_file_chunks(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    si.build_index("fake-repo", _sample_chunks(), force=True)
    assert si.chunk_count("fake-repo") == 3

    deleted = si.delete_file_chunks("fake-repo", "src/main.py")
    assert deleted == 1
    assert si.chunk_count("fake-repo") == 2


def test_has_index(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    assert si.has_index("fake-repo") is False

    si.build_index("fake-repo", _sample_chunks(), force=True)
    assert si.has_index("fake-repo") is True


def test_force_rebuild_clears_old_data(repo_config_path: Path):
    _, si = _make_index(repo_config_path)
    si.build_index("fake-repo", _sample_chunks(), force=True)
    assert si.chunk_count("fake-repo") == 3

    si.build_index("fake-repo", _sample_chunks()[:1], force=True)
    assert si.chunk_count("fake-repo") == 1
