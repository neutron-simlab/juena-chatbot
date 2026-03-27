"""Tests for keyword search."""

from pathlib import Path

from juena.retrieval.keyword_search import keyword_search
from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager


def _make_manager(repo_config_path: Path) -> RepoManager:
    return RepoManager(load_repo_configs(repo_config_path))


def test_finds_exact_match(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    hits = keyword_search(mgr, "fake-repo", "greet")
    assert len(hits) > 0
    assert any("main.py" in h.file_path for h in hits)


def test_case_insensitive(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    hits = keyword_search(mgr, "fake-repo", "GREET")
    assert len(hits) > 0


def test_no_match(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    hits = keyword_search(mgr, "fake-repo", "zzz_nonexistent_symbol_zzz")
    assert hits == []


def test_doc_flag(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    hits = keyword_search(mgr, "fake-repo", "Fake Repo")
    doc_hits = [h for h in hits if h.is_doc]
    assert len(doc_hits) > 0


def test_max_hits_respected(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    hits = keyword_search(mgr, "fake-repo", "def", max_hits=1)
    assert len(hits) <= 1
