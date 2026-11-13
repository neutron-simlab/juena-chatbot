"""Search backed by the juena-rag service. The only search backend there is.

Presents the members `juena.tools.repo_search` uses: ``search``,
``search_many`` and the hit dictionary they return. The hit stays a dictionary
of loose keys rather than a type because the tools' JSON is what the model was
prompted against. Nothing is indexed here: an ingestion job writes the store,
not this process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from juena_core.log import get_logger

logger = get_logger(__name__)

#: How many chunks to pull before filtering on ``where``. juena-rag has no
#: arbitrary metadata filter -- it narrows by source, category and chunk kind --
#: so an ``is_doc`` filter is applied here, and over-fetching is what keeps a
#: filtered search from coming back nearly empty.
OVERFETCH = 5

#: Candidates the store retrieves before it narrows to what is returned. Wider
#: than the result set because the first pass is an approximate index scan.
CANDIDATE_FACTOR = 4


class RagVectorIndex:
    """The read half of juena-rag.

    Holds no embedder and no database driver -- the service does both -- so the
    only configuration it needs is where that service is.
    """

    def __init__(self, url: str, *, timeout: float | None = None) -> None:
        """Initialize the client.

        Args:
            url: Where juena-rag is, e.g. ``http://juena-rag:8000``.
            timeout: Request timeout in seconds; None uses the client's default.

        Raises:
            ImportError: The juena-rag client package is not installed.
        """
        try:
            from juena_rag.store.http import DEFAULT_TIMEOUT, HttpStore
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "searching pgvector needs the juena-rag client, which is not "
                f"installed: {error}. `pip install juena-rag`."
            ) from None

        self._store = HttpStore(
            url, timeout=DEFAULT_TIMEOUT if timeout is None else timeout
        )
        self._url = url

    # ----------------------------------------------------------------- #
    # What `repo_search` calls
    # ----------------------------------------------------------------- #

    def search(
        self,
        repo_id: str,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """The chunks closest to ``query`` in one repository.

        ``query_embedding`` is accepted and ignored: the service embeds, which is
        what stops two processes from embedding with two different models.

        Args:
            repo_id: The repository to search.
            query: The search text.
            n_results: Maximum number of hits.
            where: Equality filter on hit fields, e.g. ``{"is_doc": True}``.
            query_embedding: Ignored.

        Returns:
            Hit dictionaries, in the store's ranking.
        """
        return self._search([repo_id], query, n_results=n_results, where=where)

    def search_many(
        self,
        repo_ids: Sequence[str],
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The same, across several repositories, ranked together.

        One request rather than one per repository: the store already ranks
        across sources, so there is no fan-out to do here.

        Args:
            repo_ids: The repositories to search.
            query: The search text.
            n_results: Maximum number of hits.
            where: Equality filter on hit fields, e.g. ``{"is_doc": True}``.

        Returns:
            Hit dictionaries, in the store's ranking; empty when no repository
            is given.
        """
        if not repo_ids:
            return []
        return self._search(list(repo_ids), query, n_results=n_results, where=where)

    @property
    def repo_ids(self) -> list[str]:
        """Which repositories the store holds. The catalogue, not a local config."""
        return [source.source_id for source in self._store.list_sources()]

    def collection_count_existing(self, repo_id: str) -> int | None:
        """Count the chunks the store holds for one repository.

        One request per call rather than a cached catalogue: the only caller runs
        it once per repository at startup, and a cache that outlives an ingest is
        how a rebuilt corpus goes on looking empty.

        Args:
            repo_id: The repository to count.

        Returns:
            The chunk count, or ``None`` if the store lacks the repository.
        """
        for source in self._store.list_sources():
            if source.source_id == repo_id:
                return source.chunk_count
        return None

    def catalogue(self) -> list[dict[str, Any]]:
        """List every repository the store holds, with how many chunks each has.

        One request rather than ``repo_ids`` plus a count per repository.

        Returns:
            One ``{"repo_id", "chunks"}`` dictionary per repository.
        """
        return [
            {"repo_id": source.source_id, "chunks": source.chunk_count}
            for source in self._store.list_sources()
        ]

    def indexed_revisions(self) -> dict[str, str]:
        """Return the commit each repository was indexed at.

        What the clones are checked out at, so the file the model opens is the
        file the search read -- not the branch head, which moves on between
        ingests.

        Returns:
            The indexed commit by repository id.
        """
        return {
            source.source_id: source.active_revision
            for source in self._store.list_sources()
        }

    def ping(self) -> None:
        """Reach the service. Raises if it or its database cannot be reached."""
        self._store.ping()

    # ----------------------------------------------------------------- #
    # The one place a juena-rag hit becomes a chatbot hit
    # ----------------------------------------------------------------- #

    def _search(
        self,
        sources: list[str],
        query: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        wanted = n_results * OVERFETCH if where else n_results
        result = self._store.search(
            query,
            sources=sources,
            k=max(wanted * CANDIDATE_FACTOR, 100),
            n=wanted,
        )

        hits = [_as_chatbot_hit(hit) for hit in result.hits]
        if where:
            hits = [hit for hit in hits if _matches(hit, where)]
        return hits[:n_results]


def _as_chatbot_hit(hit: Any) -> dict[str, Any]:
    """One ``SearchHit`` in the shape ``_serialize_vector_hit`` expects.

    Keep these keys stable: the tools' JSON is what the model was prompted
    against.
    """
    extra = hit.extra or {}
    return {
        "id": hit.chunk_id,
        "repo_id": hit.source_id,
        "file_path": hit.source_uri or "",
        "chunk_index": hit.chunk_index,
        "start_index": extra.get("start_index", 0),
        "end_index": extra.get("end_index", 0),
        "is_doc": extra.get("is_doc", False),
        "distance": hit.distance,
        "content": hit.text,
        # A line range a citation can show a person, and the `def`/`class` the
        # chunk sits inside.
        "start_line": extra.get("start_line"),
        "end_line": extra.get("end_line"),
        "symbol": extra.get("symbol"),
    }


def _matches(hit: dict[str, Any], where: dict[str, Any]) -> bool:
    """Apply a ``where`` filter, for the equality filters this codebase uses.

    Only ``{"is_doc": True}`` is passed today. Operator filters such as
    ``$and`` or ``$in`` raise rather than being ignored, since silently
    returning unfiltered results is the kind of bug nobody notices.
    """
    for key, expected in where.items():
        if isinstance(expected, dict):
            raise NotImplementedError(
                f"juena-rag has no operator filter; {key}={expected!r} cannot be "
                "translated. Narrow by source or chunk kind instead."
            )
        if hit.get(key) != expected:
            return False
    return True


def build_search_index() -> RagVectorIndex:
    """Build the search backend, which is juena-rag and nothing else.

    There is no fallback to a local index: a stale local index that keeps
    answering is the failure nobody notices.

    Returns:
        A client for the juena-rag service at ``JUENA_RAG_URL``.

    Raises:
        RuntimeError: ``JUENA_RAG_URL`` is not set.
    """
    from juena.core.config import global_config

    url = global_config.JUENA_RAG_URL
    if not url:
        raise RuntimeError(
            "JUENA_RAG_URL is not set, and repository search has no local index "
            "to fall back to -- juena-rag holds the vectors. Point it at the "
            "service: http://host.docker.internal:8100 from a container, "
            "http://127.0.0.1:8100 natively."
        )

    logger.info("Repository search is served by juena-rag at %s", url)
    return RagVectorIndex(url)
