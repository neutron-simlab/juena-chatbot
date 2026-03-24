"""Tests for hybrid (RRF) search."""

from pathlib import Path

import pytest

from juena.retrieval.hybrid_search import hybrid_search
from juena.retrieval.repo_config import load_repo_configs
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval import vector_index as vector_index_module
from juena.retrieval.vector_index import RepoVectorIndex


@pytest.fixture(autouse=True)
def _stub_embedding_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubConfig:
        BLABLADOR_API_KEY = None
        BLABLADOR_BASE_URL = None
        BLABLADOR_EMBEDDING_MODEL = "alias-qwen3-8b-embeddings"

    monkeypatch.setattr(vector_index_module, "_get_config", lambda: StubConfig)


def _setup(repo_config_path: Path):
    mgr = RepoManager(load_repo_configs(repo_config_path))
    vi = RepoVectorIndex(mgr)
    vi.build_index("fake-repo", force=True)
    return mgr, vi


def test_hybrid_returns_results(repo_config_path: Path):
    mgr, vi = _setup(repo_config_path)
    hits = hybrid_search(mgr, vi, "fake-repo", "greet", max_results=5)
    assert len(hits) > 0


def test_hybrid_rrf_score_ordering(repo_config_path: Path):
    mgr, vi = _setup(repo_config_path)
    hits = hybrid_search(mgr, vi, "fake-repo", "greet", max_results=5)
    scores = [h.rrf_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_source_label(repo_config_path: Path):
    mgr, vi = _setup(repo_config_path)
    hits = hybrid_search(mgr, vi, "fake-repo", "greet", max_results=10)
    for h in hits:
        assert h.source in ("keyword", "semantic", "both")
