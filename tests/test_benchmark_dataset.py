"""Tests for benchmark dataset generation."""

from pathlib import Path

import yaml

from juena.retrieval.benchmark_dataset import (
    BenchmarkQuery,
    generate_queries,
    write_gold_yaml,
)
from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager


def _make_manager(repo_config_path: Path) -> RepoManager:
    return RepoManager(load_repo_configs(repo_config_path))


def test_generate_queries_from_fake_repo(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"])
    assert len(queries) > 0
    assert all(isinstance(q, BenchmarkQuery) for q in queries)
    assert all(q.repo_id == "fake-repo" for q in queries)
    assert all(q.label_source == "auto" for q in queries)


def test_generate_queries_includes_python_symbols(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"])
    symbol_queries = [q for q in queries if q.query_type == "symbol"]
    assert len(symbol_queries) > 0
    greet_queries = [q for q in symbol_queries if "greet" in q.query.lower()]
    assert len(greet_queries) > 0


def test_generate_queries_includes_doc_headings(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"])
    doc_queries = [q for q in queries if q.query_type == "doc"]
    assert len(doc_queries) > 0


def test_generate_queries_includes_file_queries(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"])
    file_queries = [q for q in queries if q.query_type == "file"]
    assert len(file_queries) > 0


def test_write_gold_yaml(repo_config_path: Path, tmp_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"])
    out = tmp_path / "gold.yaml"
    write_gold_yaml(queries, out, merge=False)
    assert out.exists()

    with open(out) as f:
        data = yaml.safe_load(f)
    assert len(data["queries"]) == len(queries)


def test_write_gold_yaml_preserves_human_entries(tmp_path: Path):
    out = tmp_path / "gold.yaml"

    human_entry = {
        "repo_id": "my-repo",
        "query": "Human curated question",
        "relevant_files": ["src/foo.py"],
        "label_source": "human",
        "query_type": "conceptual",
        "difficulty": "hard",
        "notes": "Hand-written",
    }
    with open(out, "w") as f:
        yaml.dump({"queries": [human_entry]}, f)

    auto_queries = [
        BenchmarkQuery(
            repo_id="my-repo",
            query="Auto question",
            relevant_files=["src/bar.py"],
        )
    ]
    write_gold_yaml(auto_queries, out, merge=True)

    with open(out) as f:
        data = yaml.safe_load(f)

    labels = [q["label_source"] for q in data["queries"]]
    assert "human" in labels
    assert "auto" in labels
    assert len(data["queries"]) == 2


def test_max_per_repo_respected(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    queries = generate_queries(mgr, ["fake-repo"], max_per_repo=2)
    assert len(queries) <= 2
