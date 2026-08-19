"""Search backed by the juena-rag service. The only search backend there is.

Presents the members `juena.tools.repo_search` uses -- ``search``,
``search_many`` and the hit dictionary they return -- which is how it replaced
:class:`~juena.indexing.vector_index.RepoVectorIndex` without the tools noticing.
That class is retired but still on disk for the benchmark; nothing in the running
application reaches it. Indexing is not here either: the store is written by an
ingestion job, not by this process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from juena.core.log import get_logger

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
    """The read half of juena-rag, shaped like ``RepoVectorIndex``.

    Holds no embedder and no database driver -- the service does both -- so the
    only configuration it needs is where that service is.
    """

    def __init__(self, url: str, *, timeout: float | None = None) -> None:
        """
        :param url: where juena-rag is, e.g. ``http://juena-rag:8000``.
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
        across sources, so the fan-out ``RepoVectorIndex`` needs has nothing to do
        here.
        """
        if not repo_ids:
            return []
        return self._search(list(repo_ids), query, n_results=n_results, where=where)

    @property
    def repo_ids(self) -> list[str]:
        """Which repositories the store holds. The catalogue, not a local config."""
        return [source.source_id for source in self._store.list_sources()]

    def collection_count_existing(self, repo_id: str) -> int | None:
        """How many chunks this repository has, or ``None`` if the store lacks it.

        One request per call rather than a cached catalogue: the only caller runs
        it once per repository at startup, and a cache that outlives an ingest is
        how a rebuilt corpus goes on looking empty.
        """
        for source in self._store.list_sources():
            if source.source_id == repo_id:
                return source.chunk_count
        return None

    def catalogue(self) -> list[dict[str, Any]]:
        """Every repository the store holds, with how many chunks each has.

        One request rather than ``repo_ids`` plus a count per repository, which
        was nine round trips to answer one question.
        """
        return [
            {"repo_id": source.source_id, "chunks": source.chunk_count}
            for source in self._store.list_sources()
        ]

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

    The first six keys are the existing contract and must not drift -- the tools'
    JSON is what the model was prompted against. ``start_line``, ``end_line`` and
    ``symbol`` are new, and are additions rather than changes.
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
        # New with pgvector: a line range a citation can show a person, and the
        # `def`/`class` the chunk sits inside.
        "start_line": extra.get("start_line"),
        "end_line": extra.get("end_line"),
        "symbol": extra.get("symbol"),
    }


def _matches(hit: dict[str, Any], where: dict[str, Any]) -> bool:
    """Chroma's ``where`` clause, for the equality filters this codebase uses.

    Only ``{"is_doc": True}`` is passed today. Anything richer -- Chroma's
    ``$and``, ``$in`` and friends -- is refused loudly rather than ignored, since
    silently returning unfiltered results is the kind of bug nobody notices.
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
    """The search backend, which is juena-rag and nothing else.

    Refuses to start rather than falling back to the retired Chroma index: an
    unset variable used to mean "serve from whatever is on disk", and a stale
    local index that keeps answering is the failure nobody notices.
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
