"""
Hybrid search -- vector-based retrieval behind the legacy hybrid API name.

The sparse (FTS5) retriever has been removed.  This module preserves the
``hybrid_search`` entry-point for API compatibility while delegating
entirely to the vector index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from juena.core.log import get_logger
from juena.indexing.vector_index import RepoVectorIndex

logger = get_logger(__name__)


@dataclass
class HybridHit:
    file_path: str
    content: str
    is_doc: bool
    semantic_rank: int | None


def hybrid_search(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    *,
    max_results: int = 10,
) -> List[HybridHit]:
    """
    Vector-based search (compatibility wrapper).

    The ``hybrid_search`` name is kept for API compatibility.  Results are
    produced solely from the vector index.
    """
    sem_hits = vector_index.search(repo_id, query, n_results=max_results)

    results: list[HybridHit] = []
    for rank, hit in enumerate(sem_hits):
        results.append(HybridHit(
            file_path=hit["file_path"],
            content=hit["content"],
            is_doc=hit.get("is_doc", False),
            semantic_rank=rank,
        ))

    return results
