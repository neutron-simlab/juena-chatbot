"""The pgvector-backed index, asserted against the shape `repo_search` expects.

The tools were written against a hit dictionary and two method signatures, and
they still are; this file is what keeps that contract from drifting now that the
Chroma implementation it was copied from has been deleted. Nothing here needs a
network: the store is a fake.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from juena.indexing.rag_index import RagVectorIndex, _as_chatbot_hit, _matches
from juena.tools.repo_search import _serialize_vector_hit

#: What `repo_search` reads out of a hit -- `_serialize_vector_hit` for what the
#: model sees, `_distance_sort_key` for ranking across repositories. A key
#: missing here is a key the tools would silently default, which is how a hit
#: shape drifts. `id`, `start_index` and `end_index` are still produced but no
#: longer read by anything, so they are not asserted.
REQUIRED_KEYS = (
    "file_path",
    "chunk_index",
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
# The shape the tools read
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_hit_carries_every_key_the_tools_read(key):
    assert key in _as_chatbot_hit(FakeHit())


def test_a_hit_survives_the_tools_own_serialiser():
    hit = _as_chatbot_hit(FakeHit())

    serialised = _serialize_vector_hit(hit["repo_id"], hit)

    assert serialised["path"] == "/repos/jscatter/src/jscatter/dataarray.py"
    assert serialised["content"] == "def fit(self):"
    assert serialised["distance"] == 0.25
    # Dropped: the model never used them and they were a third of the payload.
    for gone in ("id", "file_path", "chunk_index", "start_index", "end_index", "char_count"):
        assert gone not in serialised


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


def test_an_unset_url_refuses_rather_than_serving_a_stale_local_index(monkeypatch):
    """The old fallback would have answered from a retired Chroma index.

    Silently serving stale results is the failure nobody notices, so a missing
    variable has to stop the process instead.
    """
    import pytest

    from juena.core import config as config_module
    from juena.indexing import rag_index

    monkeypatch.setattr(
        config_module.global_config, "JUENA_RAG_URL", None, raising=False
    )

    with pytest.raises(RuntimeError, match="JUENA_RAG_URL is not set"):
        rag_index.build_search_index()


def test_the_backend_is_juena_rag_when_a_url_is_configured(monkeypatch):
    from juena.core import config as config_module
    from juena.indexing import rag_index

    monkeypatch.setattr(
        config_module.global_config, "JUENA_RAG_URL", "http://rag:8000", raising=False
    )
    monkeypatch.setattr(rag_index, "RagVectorIndex", lambda url: f"rag at {url}")

    assert rag_index.build_search_index() == "rag at http://rag:8000"


def test_the_catalogue_reports_every_repository_with_its_chunk_count(index):
    """One request, not `repo_ids` plus a count per repository."""
    assert index.catalogue() == [
        {"repo_id": "jscatter", "chunks": 1641},
        {"repo_id": "datreat", "chunks": 12362},
        {"repo_id": "declared-only", "chunks": 0},
    ]


def test_the_repository_list_says_what_each_one_is_for(index):
    """The store knows what is searchable; the config knows what it is."""
    import json

    from juena.tools.repo_search import _list_indexed_repositories

    class StubRepoManager:
        def get_config(self, repo_id):
            if repo_id == "jscatter":
                return SimpleNamespace(name="jscatter", description="SAS analysis")
            return None

    listed = json.loads(_list_indexed_repositories(StubRepoManager(), index))

    assert listed[0] == {
        "repo_id": "jscatter",
        "chunks": 1641,
        "name": "jscatter",
        "description": "SAS analysis",
    }
    # Indexed but undescribed is reported, not hidden -- it is still searchable.
    assert listed[1]["repo_id"] == "datreat"
    assert "not described" in listed[1]["description"]


def test_the_repository_list_is_the_store_not_the_clone_directory(index):
    """`ls /repos` shows what is cloned, which is how a removed repo gets named."""
    import json

    from juena.tools.repo_search import _list_indexed_repositories

    class StubRepoManager:
        def get_config(self, repo_id):
            return SimpleNamespace(name=repo_id, description="")

    listed = json.loads(_list_indexed_repositories(StubRepoManager(), index))

    assert [row["repo_id"] for row in listed] == [
        "jscatter",
        "datreat",
        "declared-only",
    ]
