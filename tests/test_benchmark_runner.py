"""Tests for the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from juena.retrieval import benchmark_runner


@dataclass
class _FakeHybridHit:
    file_path: str
    content: str
    is_doc: bool
    keyword_rank: int | None
    semantic_rank: int | None
    rrf_score: float
    source: str


@dataclass
class _FakeSparseHit:
    file_path: str
    chunk_index: int
    content: str
    is_doc: bool
    bm25_rank: float


class StubRepoManager:
    repo_ids = ["test-repo"]


class StubVectorIndex:
    def search(self, repo_id: str, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        return [
            {"file_path": "src/main.py", "chunk_index": 0, "content": "def main()", "is_doc": False, "distance": 0.1},
            {"file_path": "README.md", "chunk_index": 0, "content": "# Project", "is_doc": True, "distance": 0.5},
        ]


class StubSparseIndex:
    def search(self, repo_id: str, query: str, *, max_results: int = 15, docs_only: bool = False) -> list:
        return [
            _FakeSparseHit("src/main.py", 0, "def main()", False, -1.0),
            _FakeSparseHit("src/utils.py", 0, "def helper()", False, -0.5),
        ]


@pytest.fixture()
def gold_yaml(tmp_path: Path) -> Path:
    data = {
        "queries": [
            {
                "repo_id": "test-repo",
                "query": "Where is the main function defined?",
                "relevant_files": ["src/main.py"],
                "label_source": "auto",
                "query_type": "symbol",
                "difficulty": "easy",
            },
            {
                "repo_id": "test-repo",
                "query": "What does the project do?",
                "relevant_files": ["README.md"],
                "label_source": "auto",
                "query_type": "doc",
                "difficulty": "easy",
            },
        ]
    }
    p = tmp_path / "gold_queries.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


def test_load_gold_queries(gold_yaml: Path):
    queries = benchmark_runner._load_gold_queries(gold_yaml)
    assert len(queries) == 2
    assert queries[0]["query"] == "Where is the main function defined?"


def test_load_gold_queries_missing_file(tmp_path: Path):
    queries = benchmark_runner._load_gold_queries(tmp_path / "nonexistent.yaml")
    assert queries == []


def test_run_dense():
    vi = StubVectorIndex()
    result = benchmark_runner._run_dense(vi, "test-repo", "main function", 10)
    assert "src/main.py" in result
    assert len(result) == len(set(result))


def test_run_sparse():
    si = StubSparseIndex()
    result = benchmark_runner._run_sparse(si, "test-repo", "main function", 10)
    assert "src/main.py" in result


def test_recall_at_k():
    results = [
        benchmark_runner.QueryResult(
            query_id=1,
            query_text="test",
            repo_id="r",
            relevant_files=["a.py", "b.py"],
            retrieved_files=["a.py", "c.py", "b.py"],
            retriever="dense",
            latency_ms=1.0,
        )
    ]
    assert benchmark_runner._recall_at_k(results, 1) == 0.5
    assert benchmark_runner._recall_at_k(results, 3) == 1.0
    assert benchmark_runner._recall_at_k([], 5) == 0.0


def test_compute_trec_metrics():
    results = [
        benchmark_runner.QueryResult(
            query_id=1,
            query_text="test",
            repo_id="r",
            relevant_files=["a.py"],
            retrieved_files=["a.py", "b.py"],
            retriever="dense",
            latency_ms=1.0,
        ),
    ]
    metrics = benchmark_runner._compute_trec_metrics(results)
    assert "map" in metrics
    assert "recip_rank" in metrics
    assert metrics["map"] == 1.0
    assert metrics["recip_rank"] == 1.0


def test_run_benchmark_with_stubs(gold_yaml: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        benchmark_runner,
        "hybrid_search",
        lambda rm, vi, repo_id, query, max_results=10, sparse_index=None: [
            _FakeHybridHit("src/main.py", "def main()", False, 0, 0, 1.0, "both"),
        ],
    )

    reports = benchmark_runner.run_benchmark(
        StubRepoManager(),
        StubVectorIndex(),
        StubSparseIndex(),
        gold_path=gold_yaml,
        k=5,
        retrievers=["dense", "sparse", "hybrid"],
    )
    assert len(reports) == 3
    for r in reports:
        assert r.num_queries == 2
        assert "map" in r.metrics
        assert "R@5" in r.metrics
        assert r.metrics["avg_latency_ms"] >= 0


def test_write_report(tmp_path: Path):
    reports = [
        benchmark_runner.BenchmarkReport(
            retriever="dense",
            num_queries=1,
            metrics={"map": 1.0, "recip_rank": 1.0, "NDCG@5": 1.0, "NDCG@10": 1.0, "P@5": 0.2, "P@10": 0.1, "R@1": 1.0, "R@3": 1.0, "R@5": 1.0, "R@10": 1.0, "avg_latency_ms": 5.2},
            per_query=[],
        ),
    ]
    out_dir = benchmark_runner.write_report(reports, output_dir=tmp_path)
    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()

    md_content = (out_dir / "benchmark_summary.md").read_text()
    assert "dense" in md_content
    assert "map" in md_content
