"""
LangChain tools for code-chat: hybrid, semantic, and docs retrieval over
configured repositories.
"""

from __future__ import annotations

import json

from langchain.tools import tool

from juena.retrieval.hybrid_search import hybrid_search
from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex

_PREVIEW_CHARS = 240


def _compact_preview(text: str, *, max_chars: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and bound long snippets for tool responses."""

    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _repo_virtual_path(repo_id: str, rel_path: str) -> str:
    """Map a repo-relative file path onto the Deep Agents /repos tree."""

    return f"/repos/{repo_id}/{rel_path.lstrip('/')}"


def _serialize_vector_hit(repo_id: str, hit: dict) -> dict:
    """Drop full chunk bodies from vector hits and keep a bounded preview."""

    file_path = str(hit.get("file_path", ""))
    content = str(hit.get("content", ""))
    return {
        "id": hit.get("id", ""),
        "repo_id": repo_id,
        "file_path": file_path,
        "path": _repo_virtual_path(repo_id, file_path),
        "chunk_index": hit.get("chunk_index", 0),
        "is_doc": hit.get("is_doc", False),
        "distance": hit.get("distance"),
        "char_count": len(content),
        "preview": _compact_preview(content),
    }


def _search_code_semantic(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    n_results: int = 10,
) -> str:
    hits = vector_index.search(repo_id, query, n_results=n_results)
    return json.dumps([_serialize_vector_hit(repo_id, hit) for hit in hits], indent=2)


def _search_code_hybrid(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    max_results: int = 10,
) -> str:
    hits = hybrid_search(
        vector_index,
        repo_id,
        query,
        max_results=max_results,
    )
    return json.dumps(
        [
            {
                "repo_id": repo_id,
                "file_path": h.file_path,
                "path": _repo_virtual_path(repo_id, h.file_path),
                "is_doc": h.is_doc,
                "semantic_rank": h.semantic_rank,
                "char_count": len(h.content),
                "preview": _compact_preview(h.content),
            }
            for h in hits
        ],
        indent=2,
    )


def _search_docs_local(
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    max_results: int = 10,
) -> str:
    hits = vector_index.search(
        repo_id,
        query,
        n_results=max_results,
        where={"is_doc": True},
    )
    return json.dumps([_serialize_vector_hit(repo_id, hit) for hit in hits], indent=2)


def build_code_chat_tools(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
) -> list:
    """Create retrieval tools bound to the prepared repository resources."""

    @tool
    def search_code_semantic(
        repo_id: str,
        query: str,
        n_results: int = 10,
    ) -> str:
        """Semantic (embedding) search in a repository's indexed code and docs.

        Use this for conceptual questions ("how does authentication work?") rather
        than exact-string lookups.

        Args:
            repo_id: Repository identifier visible under /repos/<repo_id>/.
            query: Natural-language question or concept.
            n_results: Maximum number of results.

        Returns a JSON array of hits with repo_id, file_path, path, chunk_index,
        is_doc, distance, char_count, and preview fields.
        """
        return _search_code_semantic(vector_index, repo_id, query, n_results=n_results)

    @tool
    def search_code_hybrid(
        repo_id: str,
        query: str,
        max_results: int = 10,
    ) -> str:
        """Default search tool -- semantic embedding search over the indexed
        repositories.

        Use this as the primary search for all repository questions.  It works
        for conceptual queries, symbol lookups, and everything in between.

        Args:
            repo_id: Repository identifier visible under /repos/<repo_id>/.
            query: Search query - works for both exact strings and concepts.
            max_results: Maximum results to return.

        Returns a JSON array of hits with repo_id, file_path, path, preview,
        is_doc, semantic_rank, and char_count fields.
        """
        return _search_code_hybrid(
            vector_index,
            repo_id,
            query,
            max_results=max_results,
        )

    @tool
    def search_docs_local(
        repo_id: str,
        query: str,
        max_results: int = 10,
    ) -> str:
        """Search only the documentation files (README, docs/, guides/) of a repo.

        This is a semantic search restricted to chunks tagged as documentation.

        Args:
            repo_id: Repository identifier visible under /repos/<repo_id>/.
            query: Natural-language question.
            max_results: Maximum results.

        Returns a JSON array of hits with repo_id, file_path, path, chunk_index,
        is_doc, distance, char_count, and preview fields.
        """
        return _search_docs_local(vector_index, repo_id, query, max_results=max_results)

    return [
        search_code_semantic,
        search_code_hybrid,
        search_docs_local,
    ]
