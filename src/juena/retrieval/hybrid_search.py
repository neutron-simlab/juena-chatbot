"""
Hybrid search – merges keyword hits and semantic (vector) hits into a
single ranked result list using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from juena.core.log import get_logger
from juena.retrieval.keyword_search import KeywordHit, keyword_search
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex

logger = get_logger(__name__)

RRF_K = 60  # standard RRF constant


@dataclass
class HybridHit:
    file_path: str
    content: str
    is_doc: bool
    keyword_rank: int | None
    semantic_rank: int | None
    rrf_score: float
    source: str  # "keyword", "semantic", or "both"


def hybrid_search(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    *,
    max_results: int = 10,
    keyword_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> List[HybridHit]:
    """
    Run both keyword and semantic search, then merge via RRF.
    """
    kw_hits = keyword_search(repo_manager, repo_id, query, max_hits=max_results * 2)
    sem_hits = vector_index.search(repo_id, query, n_results=max_results * 2)

    # Build per-file_path scores
    scores: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(kw_hits):
        key = f"{hit.file_path}::{hit.line_number}"
        entry = scores.setdefault(key, {
            "file_path": hit.file_path,
            "content": hit.snippet,
            "is_doc": hit.is_doc,
            "keyword_rank": None,
            "semantic_rank": None,
            "rrf": 0.0,
            "source": set(),
        })
        entry["keyword_rank"] = rank
        entry["rrf"] += keyword_weight / (RRF_K + rank + 1)
        entry["source"].add("keyword")

    for rank, hit in enumerate(sem_hits):
        key = f"{hit['file_path']}::{hit['chunk_index']}"
        entry = scores.setdefault(key, {
            "file_path": hit["file_path"],
            "content": hit["content"],
            "is_doc": hit.get("is_doc", False),
            "keyword_rank": None,
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
            keyword_rank=s["keyword_rank"],
            semantic_rank=s["semantic_rank"],
            rrf_score=s["rrf"],
            source=source_label,
        ))

    return results
