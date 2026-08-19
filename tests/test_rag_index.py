"""The pgvector-backed index, asserted against the shape it has to replace.

`RagVectorIndex` is swapped in for `RepoVectorIndex` by changing a constructor
line, which only works if the two agree on the two methods `repo_search` calls
and on the keys of the hits they return. Nothing here needs a network: the store
is a fake, and the shape checks read signatures.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any

import pytest

from juena.indexing.rag_index import RagVectorIndex, _as_chatbot_hit, _matches
from juena.indexing.vector_index import RepoVectorIndex
from juena.tools.repo_search import _serialize_vector_hit

#: What `_serialize_vector_hit` reads out of a hit. A key missing here is a key
#: the tools would silently default, which is how a hit shape drifts.
REQUIRED_KEYS = (
    "id",
    "file_path",
    "chunk_index",
    "start_index",
    "end_index",
    "is_doc",
    "distance",
    "content",
)


@dataclass
class FakeHit:
    chunk_id: str = "abc123"
    source_id: str = "jscatter"
    source_uri: str | None = "src/jscatter/dataarray.py"
    chunk_index: int = 7
    distance: float = 0.25
    text: str = "def fit(self):"
    extra: dict[str, Any] = field(
        default_factory=lambda: {
            "start_index": 100,
            "end_index": 400,
            "is_doc": False,
            "start_line": 12,
            "end_line": 30,
            "symbol": "dataArray.fit",
        }
    )


@dataclass
class FakeSource:
    source_id: str
    chunk_count: int


@dataclass
class FakeResult:
    hits: list[FakeHit]


class FakeStore:
    """Records what it was asked for, so the k/n arithmetic is checkable."""

    def __init__(self, hits: list[FakeHit] | None = None) -> None:
        self.hits = hits if hits is not None else [FakeHit()]
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> FakeResult:
        self.calls.append({"query": query, **kwargs})
        return FakeResult(self.hits)

    def list_sources(self):
        return [
            FakeSource(source_id="jscatter", chunk_count=1641),
            FakeSource(source_id="datreat", chunk_count=12362),
            FakeSource(source_id="declared-only", chunk_count=0),
        ]


@pytest.fixture
def index() -> RagVectorIndex:
    built = RagVectorIndex.__new__(RagVectorIndex)
    built._store = FakeStore()
    built._url = "http://testserver"
    return built


# --------------------------------------------------------------------------- #
# The shape -- what makes the swap a constructor line
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["search", "search_many"])
def test_the_two_indexes_take_the_same_arguments(name):
    chroma = inspect.signature(getattr(RepoVectorIndex, name)).parameters
    rag = inspect.signature(getattr(RagVectorIndex, name)).parameters

    assert list(chroma) == list(rag)
    assert [p.default for p in chroma.values()] == [p.default for p in rag.values()]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_hit_carries_every_key_the_tools_read(key):
    assert key in _as_chatbot_hit(FakeHit())


def test_a_hit_survives_the_tools_own_serialiser():
    hit = _as_chatbot_hit(FakeHit())

    serialised = _serialize_vector_hit(hit["repo_id"], hit)

    assert serialised["file_path"] == "src/jscatter/dataarray.py"
    assert serialised["path"] == "/repos/jscatter/src/jscatter/dataarray.py"
    assert serialised["preview"] == "def fit(self):"
    assert serialised["distance"] == 0.25


def test_the_locator_fields_are_added_and_not_substituted():
    """New keys are fine; changing the old ones would change model behaviour."""
    hit = _as_chatbot_hit(FakeHit())

    assert (hit["start_line"], hit["end_line"]) == (12, 30)
    assert hit["symbol"] == "dataArray.fit"
    assert (hit["start_index"], hit["end_index"]) == (100, 400)


def test_a_hit_with_no_uri_becomes_an_empty_path_not_none():
    assert _as_chatbot_hit(FakeHit(source_uri=None))["file_path"] == ""


# --------------------------------------------------------------------------- #
# The behaviour
# --------------------------------------------------------------------------- #


def test_searching_one_repository_asks_for_that_source(index):
    index.search("jscatter", "form factor", n_results=3)

    call = index._store.calls[0]
    assert call["sources"] == ["jscatter"]
    assert call["n"] == 3
    assert call["k"] >= call["n"]


def test_searching_many_repositories_is_one_request_not_one_each(index):
    index.search_many(["jscatter", "datreat"], "form factor", n_results=5)

    assert len(index._store.calls) == 1
    assert index._store.calls[0]["sources"] == ["jscatter", "datreat"]


def test_searching_no_repositories_asks_nothing(index):
    assert index.search_many([], "anything") == []
    assert index._store.calls == []


def test_a_where_filter_overfetches_then_narrows(index):
    """The store cannot filter on `is_doc`, so it has to come back with spares."""
    index.search("jscatter", "guide", n_results=4, where={"is_doc": True})

    assert index._store.calls[0]["n"] > 4


def test_the_where_filter_actually_removes_non_matching_hits(index):
    index._store.hits = [FakeHit(), FakeHit(extra={"is_doc": True})]

    kept = index.search("jscatter", "guide", n_results=10, where={"is_doc": True})

    assert [hit["is_doc"] for hit in kept] == [True]


def test_an_operator_filter_is_refused_rather_than_ignored():
    """Silently returning unfiltered results is the bug nobody notices."""
    with pytest.raises(NotImplementedError, match="operator filter"):
        _matches({"is_doc": True}, {"is_doc": {"$eq": True}})


def test_the_catalogue_comes_from_the_store_not_a_local_config(index):
    assert index.repo_ids == ["jscatter", "datreat", "declared-only"]


# --------------------------------------------------------------------------- #
# The count the bootstrap check needs
# --------------------------------------------------------------------------- #


def test_the_chunk_count_comes_from_the_catalogue(index):
    assert index.collection_count_existing("jscatter") == 1641


def test_a_source_the_store_does_not_have_counts_as_none_not_zero(index):
    """`describe_bootstrap_problems` says "missing" for None and "empty" for 0."""
    assert index.collection_count_existing("nothing-here") is None
    assert index.collection_count_existing("declared-only") == 0


def test_the_backend_is_chroma_when_no_url_is_configured(monkeypatch):
    from juena.core import config as config_module
    from juena.indexing import rag_index
    from juena.indexing.vector_index import RepoVectorIndex

    monkeypatch.setattr(
        config_module.global_config, "JUENA_RAG_URL", None, raising=False
    )
    monkeypatch.setattr(rag_index, "RagVectorIndex", lambda url: "rag")

    built = rag_index.build_search_index(object())

    assert isinstance(built, RepoVectorIndex)


def test_the_backend_is_juena_rag_when_a_url_is_configured(monkeypatch):
    """Setting one variable is the whole switch -- and unsetting it is the rollback."""
    from juena.core import config as config_module
    from juena.indexing import rag_index

    monkeypatch.setattr(
        config_module.global_config, "JUENA_RAG_URL", "http://rag:8000", raising=False
    )
    monkeypatch.setattr(rag_index, "RagVectorIndex", lambda url: f"rag at {url}")

    assert rag_index.build_search_index(object()) == "rag at http://rag:8000"
