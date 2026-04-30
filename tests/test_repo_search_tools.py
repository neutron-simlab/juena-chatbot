"""Tests for repo-search tool wiring."""

from __future__ import annotations

import json
from typing import cast

from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.tools import repo_search


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
            "is_doc": False,
            "distance": 0.12,
            "char_count": len(content),
            "preview": repo_search._compact_preview(content),
        }
    ]


def test_search_code_hybrid_returns_preview_instead_of_full_content() -> None:
    content = "service layer " + ("y" * 300)

    class StubVectorIndex:
        pass

    class StubRepoManager:
        repo_ids = ["repo-a"]

    class StubHybridHit:
        def __init__(self) -> None:
            self.file_path = "src/service.py"
            self.content = content
            self.is_doc = False
            self.semantic_rank = 0

    original = repo_search.hybrid_search
    repo_search.hybrid_search = lambda *args, **kwargs: [StubHybridHit()]
    try:
        payload = json.loads(
            repo_search._search_code_hybrid(
                cast(RepoManager, StubRepoManager()),
                cast(RepoVectorIndex, StubVectorIndex()),
                "repo-a",
                "service",
                max_results=2,
            )
        )
    finally:
        repo_search.hybrid_search = original

    assert payload == [
        {
            "repo_id": "repo-a",
            "file_path": "src/service.py",
            "path": "/repos/repo-a/src/service.py",
            "is_doc": False,
            "semantic_rank": 0,
            "char_count": len(content),
            "preview": repo_search._compact_preview(content),
        }
    ]
    assert "content" not in payload[0]


def test_search_code_semantic_all_repositories_returns_globally_ranked_hits() -> None:
    class StubRepoManager:
        repo_ids = ["repo-a", "repo-b"]

    class StubVectorIndex:
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


def test_search_code_hybrid_all_repositories_returns_global_preview_hits() -> None:
    class StubRepoManager:
        repo_ids = ["repo-a", "repo-b"]

    class StubVectorIndex:
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert query == "detector"
            assert n_results == 1
            assert where is None
            return [
                {
                    "id": f"{repo_id}-hit",
                    "file_path": "src/detector.py",
                    "chunk_index": 0,
                    "is_doc": False,
                    "distance": 0.2 if repo_id == "repo-a" else 0.1,
                    "content": f"{repo_id} detector implementation",
                }
            ]

    payload = json.loads(
        repo_search._search_code_hybrid(
            cast(RepoManager, StubRepoManager()),
            cast(RepoVectorIndex, StubVectorIndex()),
            "all",
            "detector",
            max_results=1,
        )
    )

    assert payload == [
        {
            "repo_id": "repo-b",
            "file_path": "src/detector.py",
            "path": "/repos/repo-b/src/detector.py",
            "is_doc": False,
            "semantic_rank": 0,
            "char_count": len("repo-b detector implementation"),
            "preview": "repo-b detector implementation",
        }
    ]
    assert "content" not in payload[0]


def test_search_code_hybrid_unknown_repo_id_returns_recoverable_error() -> None:
    class StubRepoManager:
        repo_ids = ["datreat", "jscatter", "mdanse"]

    class StubVectorIndex:
        def search(self, *args: object, **kwargs: object) -> list[dict]:
            raise AssertionError("Unknown repo ids should not reach the vector index")

    payload = json.loads(
        repo_search._search_code_hybrid(
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

    class StubVectorIndex:
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
        "search_code_hybrid",
        "search_docs_local",
    ]
    hybrid_tool = next(tool for tool in tools if tool.name == "search_code_hybrid")
    assert "candidate paths and previews" in hybrid_tool.description
    assert "read_file" in hybrid_tool.description
    assert "grep/glob" in hybrid_tool.description
    assert 'repo_id="all"' in hybrid_tool.description
