"""
Semantic (dense vector) search over the indexed repositories.

Previously named ``hybrid_search``, from when a sparse FTS5 retriever ran
alongside the vector index. That retriever was removed; this is dense-only.
The name was corrected because callers -- and the tool descriptions shown to the
model -- were promising exact-string matching that this cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from juena.core.log import get_logger
from juena.indexing.vector_index import RepoVectorIndex

logger = get_logger(__name__)


@dataclass
class SemanticHit:
    file_path: str
    content: str
    is_doc: bool
    semantic_rank: int | None
    start_index: int
    end_index: int


def semantic_search(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    *,
    max_results: int = 10,
) -> List[SemanticHit]:
    """Rank chunks of one repository by embedding similarity to ``query``."""
    return [
        SemanticHit(
            file_path=hit["file_path"],
            content=hit["content"],
            is_doc=hit.get("is_doc", False),
            semantic_rank=rank,
            start_index=int(hit.get("start_index", 0)),
            end_index=int(hit.get("end_index", 0)),
        )
        for rank, hit in enumerate(
            vector_index.search(repo_id, query, n_results=max_results)
        )
    ]
