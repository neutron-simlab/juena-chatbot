"""
Hybrid search – merges sparse (FTS5) hits and dense (vector) hits into a
single ranked result list using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from juena.core.log import get_logger
from juena.indexing.repo_manager import RepoManager
from juena.indexing.sparse_index import RepoSparseIndex
from juena.indexing.vector_index import RepoVectorIndex

logger = get_logger(__name__)

RRF_K = 60  # standard RRF constant


@dataclass
class HybridHit:
    file_path: str
    content: str
    is_doc: bool
    sparse_rank: int | None
    semantic_rank: int | None
    rrf_score: float
    source: str  # "sparse", "semantic", or "both"


def hybrid_search(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    *,
    max_results: int = 10,
    sparse_weight: float = 1.0,
    semantic_weight: float = 1.0,
    sparse_index: RepoSparseIndex | None = None,
) -> List[HybridHit]:
    """
    Run both sparse (FTS5) and dense (vector) search, then merge via RRF.

    If *sparse_index* is not provided, a new one is created from *repo_manager*.
    """
    if sparse_index is None:
        sparse_index = RepoSparseIndex(repo_manager)

    sp_hits = sparse_index.search(repo_id, query, max_results=max_results * 2)
    sem_hits = vector_index.search(repo_id, query, n_results=max_results * 2)

    scores: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(sp_hits):
        key = f"{hit.file_path}::{hit.chunk_index}"
        entry = scores.setdefault(key, {
            "file_path": hit.file_path,
            "content": hit.content,
            "is_doc": hit.is_doc,
            "sparse_rank": None,
            "semantic_rank": None,
            "rrf": 0.0,
            "source": set(),
        })
        entry["sparse_rank"] = rank
        entry["rrf"] += sparse_weight / (RRF_K + rank + 1)
        entry["source"].add("sparse")

    for rank, hit in enumerate(sem_hits):
        key = f"{hit['file_path']}::{hit['chunk_index']}"
        entry = scores.setdefault(key, {
            "file_path": hit["file_path"],
            "content": hit["content"],
            "is_doc": hit.get("is_doc", False),
            "sparse_rank": None,
            "semantic_rank": None,
            "rrf": 0.0,
            "source": set(),
        })
        if entry["semantic_rank"] is None:
            entry["semantic_rank"] = rank
        entry["rrf"] += semantic_weight / (RRF_K + rank + 1)
        entry["source"].add("semantic")

    ranked = sorted(scores.values(), key=lambda s: -s["rrf"])

    results: list[HybridHit] = []
    for s in ranked[:max_results]:
        src = s["source"]
        source_label = "both" if len(src) > 1 else next(iter(src))
        results.append(HybridHit(
            file_path=s["file_path"],
            content=s["content"],
            is_doc=s["is_doc"],
            sparse_rank=s["sparse_rank"],
            semantic_rank=s["semantic_rank"],
            rrf_score=s["rrf"],
            source=source_label,
        ))

    return results
