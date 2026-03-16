"""
LangChain tools for code-chat: list repos, keyword / semantic / hybrid
search, file reading, and local-docs search.

A singleton ``ToolContext`` is created at import time from the config.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from juena.core.log import get_logger
from juena.retrieval.hybrid_search import hybrid_search
from juena.retrieval.keyword_search import keyword_search
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex

logger = get_logger(__name__)


class ToolContext:
    """Shared state for all repo-search tools (created once at startup)."""

    _instance: Optional["ToolContext"] = None

    def __init__(self) -> None:
        self.repo_manager = RepoManager()
        self.vector_index = RepoVectorIndex(self.repo_manager)

    @classmethod
    def get(cls) -> "ToolContext":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def _ctx() -> ToolContext:
    return ToolContext.get()


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

@tool
def list_repositories() -> str:
    """List all software repositories available for code search.

    Returns a JSON array of objects with id, name, description, and source_url.
    Call this first so you know which repo_id values are valid.
    """
    meta = _ctx().repo_manager.list_repo_metadata()
    return json.dumps(meta, indent=2)


@tool
def search_code_keyword(repo_id: str, query: str, max_hits: int = 10) -> str:
    """Keyword (grep-style) search in a repository's source code and docs.

    Args:
        repo_id: Repository identifier (from list_repositories).
        query: Search string (case-insensitive substring match).
        max_hits: Maximum number of snippet results to return.

    Returns a JSON array of hits with file_path, line_number, snippet,
    match_count, and is_doc fields.
    """
    hits = keyword_search(
        _ctx().repo_manager, repo_id, query, max_hits=max_hits,
    )
    return json.dumps(
        [
            {
                "file_path": h.file_path,
                "line_number": h.line_number,
                "snippet": h.snippet,
                "match_count": h.match_count,
                "is_doc": h.is_doc,
            }
            for h in hits
        ],
        indent=2,
    )


@tool
def search_code_semantic(repo_id: str, query: str, n_results: int = 10) -> str:
    """Semantic (embedding) search in a repository's indexed code and docs.

    Use this for conceptual questions ("how does authentication work?") rather
    than exact-string lookups.

    Args:
        repo_id: Repository identifier (from list_repositories).
        query: Natural-language question or concept.
        n_results: Maximum number of results.

    Returns a JSON array of hits with file_path, chunk_index, is_doc,
    distance, and content fields.
    """
    hits = _ctx().vector_index.search(repo_id, query, n_results=n_results)
    return json.dumps(hits, indent=2)


@tool
def search_code_hybrid(repo_id: str, query: str, max_results: int = 10) -> str:
    """Hybrid search combining keyword and semantic retrieval (recommended).

    Uses Reciprocal Rank Fusion to merge keyword grep results and embedding
    similarity results into a single ranked list.

    Args:
        repo_id: Repository identifier (from list_repositories).
        query: Search query – works for both exact strings and concepts.
        max_results: Maximum results to return.

    Returns a JSON array of hits with file_path, content, is_doc,
    keyword_rank, semantic_rank, rrf_score, and source fields.
    """
    hits = hybrid_search(
        _ctx().repo_manager,
        _ctx().vector_index,
        repo_id,
        query,
        max_results=max_results,
    )
    return json.dumps(
        [
            {
                "file_path": h.file_path,
                "content": h.content,
                "is_doc": h.is_doc,
                "keyword_rank": h.keyword_rank,
                "semantic_rank": h.semantic_rank,
                "rrf_score": round(h.rrf_score, 6),
                "source": h.source,
            }
            for h in hits
        ],
        indent=2,
    )


@tool
def read_repo_file(
    repo_id: str, path: str, start_line: int = 1, end_line: int = -1
) -> str:
    """Read a file (or a range of lines) from a repository.

    Args:
        repo_id: Repository identifier (from list_repositories).
        path: Repo-relative file path (e.g. "src/agent/config.py").
        start_line: First line to include (1-based, default 1).
        end_line: Last line to include (inclusive, -1 = end of file).

    Returns the file content with line numbers prefixed.
    """
    content = _ctx().repo_manager.read_file(repo_id, path)
    lines = content.splitlines()

    start = max(1, start_line) - 1
    end = len(lines) if end_line == -1 else min(end_line, len(lines))

    numbered = []
    for i in range(start, end):
        numbered.append(f"{i + 1:>6}| {lines[i]}")
    return "\n".join(numbered)


@tool
def search_docs_local(repo_id: str, query: str, max_results: int = 10) -> str:
    """Search only the documentation files (README, docs/, guides/) of a repo.

    This is a semantic search restricted to chunks tagged as documentation.

    Args:
        repo_id: Repository identifier (from list_repositories).
        query: Natural-language question.
        max_results: Maximum results.
    """
    hits = _ctx().vector_index.search(
        repo_id, query, n_results=max_results, where={"is_doc": True},
    )
    return json.dumps(hits, indent=2)


def ensure_indices_built() -> None:
    """Build vector indices for all repos if not already populated."""
    ctx = _ctx()
    for repo_id in ctx.repo_manager.repo_ids:
        try:
            count = ctx.vector_index.collection_count(repo_id)
            if count == 0:
                logger.info("Building index for repo %s …", repo_id)
                ctx.vector_index.build_index(repo_id)
            else:
                logger.info("Repo %s already indexed (%d chunks)", repo_id, count)
        except Exception as exc:
            logger.error("Failed to build index for %s: %s", repo_id, exc)


# Expose the tool list for agent construction
CODE_CHAT_TOOLS = [
    list_repositories,
    search_code_keyword,
    search_code_semantic,
    search_code_hybrid,
    read_repo_file,
    search_docs_local,
]
