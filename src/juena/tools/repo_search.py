"""
LangChain tools for code-chat: hybrid, semantic, and docs retrieval over
configured repositories.
"""

from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Any

from langchain.tools import tool

from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex

_PREVIEW_CHARS = 240
_ALL_REPOS_ID = "all"
_MAX_REPO_ID_SUGGESTIONS = 3


def _compact_preview(text: str, *, max_chars: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and bound long snippets for tool responses."""

    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _repo_virtual_path(repo_id: str, rel_path: str) -> str:
    """Map a repo-relative file path onto the Deep Agents /repos tree."""

    return f"/repos/{repo_id}/{rel_path.lstrip('/')}"


def _is_all_repos(repo_id: str) -> bool:
    return repo_id.strip().lower() == _ALL_REPOS_ID


def _normalise_repo_label(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _resolve_repo_id(repo_manager: RepoManager, repo_id: str) -> str | None:
    repo_id = repo_id.strip()
    repo_ids = repo_manager.repo_ids
    if repo_id in repo_ids:
        return repo_id

    lower_to_id = {candidate.lower(): candidate for candidate in repo_ids}
    if repo_id.lower() in lower_to_id:
        return lower_to_id[repo_id.lower()]

    normalised_to_id = {
        _normalise_repo_label(candidate): candidate
        for candidate in repo_ids
    }
    return normalised_to_id.get(_normalise_repo_label(repo_id))


def _unknown_repo_response(repo_manager: RepoManager, repo_id: str) -> str:
    repo_ids = repo_manager.repo_ids
    suggestions = get_close_matches(
        repo_id.strip(),
        repo_ids,
        n=_MAX_REPO_ID_SUGGESTIONS,
        cutoff=0.55,
    )
    return json.dumps(
        {
            "error": f"Unknown repo_id: {repo_id}",
            "hint": 'Use repo_id="all" when the relevant repository is unknown.',
            "available_repo_ids": repo_ids,
            "suggested_repo_ids": suggestions,
        },
        separators=(",", ":"),
    )


def _distance_sort_key(hit: dict[str, Any]) -> tuple[float, str, str, int]:
    distance = hit.get("distance")
    numeric_distance = float(distance) if isinstance(distance, (int, float)) else float("inf")
    return (
        numeric_distance,
        str(hit.get("repo_id", "")),
        str(hit.get("file_path", "")),
        int(hit.get("chunk_index", 0) or 0),
    )


def _search_repos_semantic(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    query: str,
    *,
    n_results: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search all indexed repositories and return globally ranked hits."""

    hits = vector_index.search_many(
        repo_manager.repo_ids,
        query,
        n_results=n_results,
        where=where,
    )
    hits.sort(key=_distance_sort_key)
    return hits[:n_results]


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
        "start_index": hit.get("start_index", 0),
        "end_index": hit.get("end_index", 0),
        "is_doc": hit.get("is_doc", False),
        "distance": hit.get("distance"),
        "char_count": len(content),
        "preview": _compact_preview(content),
    }


def _search_code_semantic(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    n_results: int = 10,
) -> str:
    if _is_all_repos(repo_id):
        hits = _search_repos_semantic(
            repo_manager,
            vector_index,
            query,
            n_results=n_results,
        )
        return json.dumps([_serialize_vector_hit(str(hit["repo_id"]), hit) for hit in hits], separators=(",", ":"))

    resolved_repo_id = _resolve_repo_id(repo_manager, repo_id)
    if resolved_repo_id is None:
        return _unknown_repo_response(repo_manager, repo_id)

    hits = vector_index.search(resolved_repo_id, query, n_results=n_results)
    return json.dumps([_serialize_vector_hit(resolved_repo_id, hit) for hit in hits], separators=(",", ":"))


def _search_docs_local(
    repo_manager: RepoManager,
    vector_index: RepoVectorIndex,
    repo_id: str,
    query: str,
    max_results: int = 10,
) -> str:
    if _is_all_repos(repo_id):
        hits = _search_repos_semantic(
            repo_manager,
            vector_index,
            query,
            n_results=max_results,
            where={"is_doc": True},
        )
        return json.dumps([_serialize_vector_hit(str(hit["repo_id"]), hit) for hit in hits], separators=(",", ":"))

    resolved_repo_id = _resolve_repo_id(repo_manager, repo_id)
    if resolved_repo_id is None:
        return _unknown_repo_response(repo_manager, repo_id)

    hits = vector_index.search(
        resolved_repo_id,
        query,
        n_results=max_results,
        where={"is_doc": True},
    )
    return json.dumps([_serialize_vector_hit(resolved_repo_id, hit) for hit in hits], separators=(",", ":"))


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
        """Default search tool -- semantic embedding search over the indexed
        repositories' code and docs.

        Use this as the primary search for all repository questions. It matches
        on meaning, so it is strong for conceptual queries ("how does
        authentication work?") and approximate symbol lookups, but it is NOT
        exact-string search -- use grep for literal tokens, config keys, and
        call sites. Treat results as candidate paths and previews, not final
        evidence. After a relevant hit, call read_file on the returned
        /repos/... path for exact line-numbered evidence, and use grep/glob to
        follow symbols or files. Use repo_id="all" when the relevant repository
        is unknown or multiple repositories may matter.

        Args:
            repo_id: Repository identifier visible under /repos/<repo_id>/, or
                "all" to search all indexed repositories.
            query: Search query -- matched semantically, not literally.
            n_results: Maximum number of results.

        Returns a JSON array of hits with repo_id, file_path, path, chunk_index,
        is_doc, distance, char_count, and preview fields. The preview is compact
        and should not be cited as final evidence.
        """
        return _search_code_semantic(
            repo_manager,
            vector_index,
            repo_id,
            query,
            n_results=n_results,
        )

    @tool
    def search_docs_local(
        repo_id: str,
        query: str,
        max_results: int = 10,
    ) -> str:
        """Search only the documentation files (README, docs/, guides/) of a repo.

        This is a semantic search restricted to chunks tagged as documentation.
        Use repo_id="all" when the relevant repository is unknown or when docs
        from multiple repositories may matter.

        Args:
            repo_id: Repository identifier visible under /repos/<repo_id>/, or
                "all" to search all indexed repositories.
            query: Natural-language question.
            max_results: Maximum results.

        Returns a JSON array of hits with repo_id, file_path, path, chunk_index,
        is_doc, distance, char_count, and preview fields.
        """
        return _search_docs_local(
            repo_manager,
            vector_index,
            repo_id,
            query,
            max_results=max_results,
        )

    return [
        search_code_semantic,
        search_docs_local,
    ]
