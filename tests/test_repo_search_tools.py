"""Tests for repo-search tool wiring."""

from __future__ import annotations

import json
from typing import cast

from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.tools import repo_search


class _FanOutStub:
    """Mirrors ``RepoVectorIndex.search_many`` on top of a stub's own ``search``.

    The real implementation embeds once and fans out concurrently; the ordering
    and repo_id tagging it guarantees are what these tests exercise.
    """

    def search_many(self, repo_ids, query, n_results=10, where=None):
        hits = []
        for repo_id in repo_ids:
            hits.extend(
                {**hit, "repo_id": repo_id}
                for hit in self.search(repo_id, query, n_results=n_results, where=where)
            )
        return hits



def test_search_code_semantic_returns_compact_preview() -> None:
    content = "def login(user):\n" + ("x" * 400)

    class StubRepoManager:
        repo_ids = ["repo-a"]

    class StubVectorIndex:
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert repo_id == "repo-a"
            assert query == "auth flow"
            assert n_results == 3
            assert where is None
            return [
                {
                    "id": "hit-1",
                    "file_path": "src/auth.py",
                    "chunk_index": 4,
                    "start_index": 120,
                    "end_index": 120 + len(content),
                    "is_doc": False,
                    "distance": 0.12,
                    "content": content,
                }
            ]

    payload = json.loads(
        repo_search._search_code_semantic(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "repo-a",
            "auth flow",
            n_results=3,
        )
    )

    assert payload == [
        {
            "id": "hit-1",
            "repo_id": "repo-a",
            "file_path": "src/auth.py",
            "path": "/repos/repo-a/src/auth.py",
            "chunk_index": 4,
            "start_index": 120,
            "end_index": 120 + len(content),
            "is_doc": False,
            "distance": 0.12,
            "char_count": len(content),
            "preview": repo_search._compact_preview(content),
        }
    ]


def test_search_code_semantic_all_repositories_returns_globally_ranked_hits() -> None:
    class StubRepoManager:
        repo_ids = ["repo-a", "repo-b"]

    class StubVectorIndex(_FanOutStub):
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert query == "instrument control"
            assert n_results == 2
            assert where is None
            if repo_id == "repo-a":
                return [
                    {
                        "id": "repo-a-hit",
                        "file_path": "src/control.py",
                        "chunk_index": 0,
                        "is_doc": False,
                        "distance": 0.4,
                        "content": "repo a control",
                    }
                ]
            if repo_id == "repo-b":
                return [
                    {
                        "id": "repo-b-hit",
                        "file_path": "src/instrument.py",
                        "chunk_index": 1,
                        "is_doc": False,
                        "distance": 0.1,
                        "content": "repo b instrument control",
                    }
                ]
            raise AssertionError(f"Unexpected repo id: {repo_id}")

    payload = json.loads(
        repo_search._search_code_semantic(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "all",
            "instrument control",
            n_results=2,
        )
    )

    assert [hit["repo_id"] for hit in payload] == ["repo-b", "repo-a"]
    assert [hit["path"] for hit in payload] == [
        "/repos/repo-b/src/instrument.py",
        "/repos/repo-a/src/control.py",
    ]


def test_search_code_semantic_unknown_repo_id_returns_recoverable_error() -> None:
    class StubRepoManager:
        repo_ids = ["datreat", "jscatter", "mdanse"]

    class StubVectorIndex:
        def search(self, *args: object, **kwargs: object) -> list[dict]:
            raise AssertionError("Unknown repo ids should not reach the vector index")

    payload = json.loads(
        repo_search._search_code_semantic(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "js scatter",
            "structure factor",
        )
    )

    assert payload["error"] == "Unknown repo_id: js scatter"
    assert payload["hint"] == 'Use repo_id="all" when the relevant repository is unknown.'
    assert payload["available_repo_ids"] == ["datreat", "jscatter", "mdanse"]
    assert "jscatter" in payload["suggested_repo_ids"]


def test_search_code_semantic_normalises_repo_id_labels() -> None:
    class StubRepoManager:
        repo_ids = ["jscatter"]

    class StubVectorIndex:
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert repo_id == "jscatter"
            assert query == "structure"
            assert n_results == 1
            assert where is None
            return [
                {
                    "id": "hit",
                    "file_path": "README.md",
                    "chunk_index": 0,
                    "is_doc": True,
                    "distance": 0.1,
                    "content": "jscatter docs",
                }
            ]

    payload = json.loads(
        repo_search._search_code_semantic(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "JScatter",
            "structure",
            n_results=1,
        )
    )

    assert payload[0]["repo_id"] == "jscatter"


def test_search_docs_local_all_repositories_restricts_to_doc_hits() -> None:
    class StubRepoManager:
        repo_ids = ["repo-a"]

    class StubVectorIndex(_FanOutStub):
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert repo_id == "repo-a"
            assert query == "setup"
            assert n_results == 1
            assert where == {"is_doc": True}
            return [
                {
                    "id": "doc-hit",
                    "file_path": "README.md",
                    "chunk_index": 0,
                    "is_doc": True,
                    "distance": 0.2,
                    "content": "setup docs",
                }
            ]

    payload = json.loads(
        repo_search._search_docs_local(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "all",
            "setup",
            max_results=1,
        )
    )

    assert payload[0]["repo_id"] == "repo-a"
    assert payload[0]["path"] == "/repos/repo-a/README.md"
    assert payload[0]["is_doc"] is True


def test_code_chat_tools_only_exposes_unique_retrieval_tools() -> None:
    tools = repo_search.build_code_chat_tools(
        cast(RepoManager, object()),
        cast(RepoVectorIndex, object()),
    )

    assert [tool.name for tool in tools] == [
        "search_code_semantic",
        "search_docs_local",
    ]
    semantic_tool = next(tool for tool in tools if tool.name == "search_code_semantic")
    assert "candidate paths and previews" in semantic_tool.description
    assert "read_file" in semantic_tool.description
    assert "grep/glob" in semantic_tool.description
    assert 'repo_id="all"' in semantic_tool.description
