"""
Benchmark runner for retrieval evaluation.

Loads gold queries from YAML, runs each query against dense (vector)
and hybrid retrievers, then computes standard IR metrics
(MAP, MRR, nDCG@K, P@K) via ``huggingface/evaluate``'s ``trec_eval``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

import yaml

from juena.core.log import get_logger
from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.retrieval.hybrid_search import hybrid_search

logger = get_logger(__name__)

_BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "retrieval"
_GOLD_PATH_DEFAULT = _BENCHMARK_DIR / "gold_queries.yaml"
_LLM_PATH_DEFAULT = _BENCHMARK_DIR / "llm-generated-queries.yaml"


@dataclass
class QueryResult:
    query_id: int
    query_text: str
    repo_id: str
    relevant_files: list[str]
    retrieved_files: list[str]
    retriever: str
    latency_ms: float


@dataclass
class BenchmarkReport:
    retriever: str
    num_queries: int
    metrics: dict[str, float]
    per_query: list[QueryResult] = field(default_factory=list)


def _load_queries_from_yaml(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Query file not found at %s", path)
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("queries", [])


def _load_gold_queries(path: Path | None = None) -> list[dict[str, Any]]:
    return _load_queries_from_yaml(path or _GOLD_PATH_DEFAULT)


def load_all_benchmark_queries(
    *,
    gold_path: Path | None = None,
    llm_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Load and merge queries from all benchmark YAML files."""
    queries: list[dict[str, Any]] = []
    queries.extend(_load_queries_from_yaml(gold_path or _GOLD_PATH_DEFAULT))
    queries.extend(_load_queries_from_yaml(llm_path or _LLM_PATH_DEFAULT))
    for p in extra_paths or []:
        queries.extend(_load_queries_from_yaml(p))
    return queries


def _run_dense(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    k: int,
) -> list[str]:
    hits = vector_index.search(repo_id, query, n_results=k)
    seen: list[str] = []
    for h in hits:
        fp = h["file_path"]
        if fp not in seen:
            seen.append(fp)
    return seen


def _run_hybrid(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    k: int,
) -> list[str]:
    hits = hybrid_search(
        vector_index, repo_id, query,
        max_results=k,
    )
    seen: list[str] = []
    for h in hits:
        if h.file_path not in seen:
            seen.append(h.file_path)
    return seen


def _compute_trec_metrics(
    query_results: list[QueryResult],
) -> dict[str, float]:
    """Compute IR metrics using ``evaluate``'s ``trec_eval``."""
    import evaluate

    if not query_results:
        return {}

    pred_queries: list[int] = []
    pred_q0: list[str] = []
    pred_docids: list[str] = []
    pred_ranks: list[int] = []
    pred_scores: list[float] = []
    pred_system: list[str] = []

    ref_queries: list[int] = []
    ref_q0: list[str] = []
    ref_docids: list[str] = []
    ref_rels: list[int] = []

    for qr in query_results:
        qid = qr.query_id

        for rank, doc_path in enumerate(qr.retrieved_files):
            pred_queries.append(qid)
            pred_q0.append("Q0")
            pred_docids.append(doc_path)
            pred_ranks.append(rank)
            pred_scores.append(float(len(qr.retrieved_files) - rank))
            pred_system.append(qr.retriever)

        relevant_set = set(qr.relevant_files)
        all_docs = set(qr.retrieved_files) | relevant_set
        for doc_path in all_docs:
            ref_queries.append(qid)
            ref_q0.append("0")
            ref_docids.append(doc_path)
            ref_rels.append(1 if doc_path in relevant_set else 0)

    if not pred_queries:
        return {}

    metric = evaluate.load("trec_eval")
    result = metric.compute(
        predictions=[{
            "query": pred_queries,
            "q0": pred_q0,
            "docid": pred_docids,
            "rank": pred_ranks,
            "score": pred_scores,
            "system": pred_system,
        }],
        references=[{
            "query": ref_queries,
            "q0": ref_q0,
            "docid": ref_docids,
            "rel": ref_rels,
        }],
    )

    cleaned: dict[str, float] = {}
    for k, v in result.items():
        if isinstance(v, str):
            continue
        try:
            cleaned[k] = float(v)
        except (TypeError, ValueError):
            continue

    return cleaned


def _recall_at_k(query_results: list[QueryResult], k: int) -> float:
    """File-level Recall@K computed manually (not in trec_eval)."""
    if not query_results:
        return 0.0
    recalls: list[float] = []
    for qr in query_results:
        relevant_set = set(qr.relevant_files)
        if not relevant_set:
            continue
        retrieved_at_k = set(qr.retrieved_files[:k])
        recalls.append(len(relevant_set & retrieved_at_k) / len(relevant_set))
    return sum(recalls) / len(recalls) if recalls else 0.0


def run_benchmark(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    *,
    gold_path: Path | None = None,
    llm_path: Path | None = None,
    extra_query_paths: list[Path] | None = None,
    k: int = 10,
    retrievers: list[str] | None = None,
) -> list[BenchmarkReport]:
    """
    Run the retrieval benchmark and return reports for each retriever.

    Parameters
    ----------
    repo_manager : RepoManager
    vector_index : RepoVectorIndex
    gold_path : optional path to gold_queries.yaml
    llm_path : optional path to llm-generated-queries.yaml
    extra_query_paths : additional YAML files to load queries from
    k : number of results to retrieve per query
    retrievers : subset of ``["dense", "hybrid"]`` to evaluate

    Returns
    -------
    list of BenchmarkReport, one per retriever
    """
    retrievers = retrievers or ["dense", "hybrid"]
    queries = load_all_benchmark_queries(
        gold_path=gold_path,
        llm_path=llm_path,
        extra_paths=extra_query_paths,
    )

    if not queries:
        logger.warning("No benchmark queries found – returning empty reports")
        return []

    indexed_repos = set(repo_manager.repo_ids)
    active_queries = [
        q for q in queries
        if q.get("repo_id") in indexed_repos
    ]

    if not active_queries:
        logger.warning("No benchmark queries match indexed repositories")
        return []

    logger.info(
        "Running benchmark: %d queries across %s retrievers",
        len(active_queries),
        retrievers,
    )

    reports: list[BenchmarkReport] = []

    for retriever_name in retrievers:
        results: list[QueryResult] = []

        for qid, q in enumerate(active_queries, start=1):
            repo_id = q["repo_id"]
            query_text = q["query"]
            relevant_files = q.get("relevant_files", [])

            t0 = time.perf_counter()

            if retriever_name == "dense":
                retrieved = _run_dense(vector_index, repo_id, query_text, k)
            elif retriever_name == "hybrid":
                retrieved = _run_hybrid(vector_index, repo_id, query_text, k)
            else:
                logger.warning("Unknown retriever %s – skipping", retriever_name)
                continue

            latency_ms = (time.perf_counter() - t0) * 1000

            results.append(QueryResult(
                query_id=qid,
                query_text=query_text,
                repo_id=repo_id,
                relevant_files=relevant_files,
                retrieved_files=retrieved,
                retriever=retriever_name,
                latency_ms=latency_ms,
            ))

        metrics = _compute_trec_metrics(results)
        for cutoff in [1, 3, 5, 10]:
            metrics[f"R@{cutoff}"] = _recall_at_k(results, cutoff)

        avg_latency = (
            sum(r.latency_ms for r in results) / len(results) if results else 0.0
        )
        metrics["avg_latency_ms"] = avg_latency

        reports.append(BenchmarkReport(
            retriever=retriever_name,
            num_queries=len(results),
            metrics=metrics,
            per_query=results,
        ))

        logger.info(
            "Retriever '%s': MAP=%.4f  MRR=%.4f  nDCG@5=%.4f  R@5=%.4f  avg_latency=%.1fms",
            retriever_name,
            metrics.get("map", 0.0),
            metrics.get("recip_rank", 0.0),
            metrics.get("NDCG@5", 0.0),
            metrics.get("R@5", 0.0),
            avg_latency,
        )

    return reports


def write_report(
    reports: list[BenchmarkReport],
    output_dir: Path | None = None,
) -> Path:
    """Write benchmark reports as JSON and a Markdown summary table."""
    output_dir = output_dir or (
        Path(__file__).resolve().parents[3] / "benchmarks" / "retrieval" / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_results.json"
    serializable = []
    for r in reports:
        serializable.append({
            "retriever": r.retriever,
            "num_queries": r.num_queries,
            "metrics": r.metrics,
            "per_query": [asdict(qr) for qr in r.per_query],
        })
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    md_path = output_dir / "benchmark_summary.md"
    key_metrics = ["map", "recip_rank", "NDCG@5", "NDCG@10", "P@5", "P@10", "R@1", "R@3", "R@5", "R@10", "avg_latency_ms"]
    header = "| Metric | " + " | ".join(r.retriever for r in reports) + " |"
    separator = "|---|" + "|".join("---" for _ in reports) + "|"

    rows: list[str] = []
    for m in key_metrics:
        vals = []
        for r in reports:
            v = r.metrics.get(m, 0.0)
            if m == "avg_latency_ms":
                vals.append(f"{v:.1f}ms")
            else:
                vals.append(f"{v:.4f}")
        rows.append(f"| {m} | " + " | ".join(vals) + " |")

    with open(md_path, "w") as f:
        f.write("# Retrieval Benchmark Results\n\n")
        f.write(f"Queries evaluated: {reports[0].num_queries if reports else 0}\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        f.write("\n".join(rows) + "\n")

    logger.info("Benchmark results written to %s", output_dir)
    return output_dir
