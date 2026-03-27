"""Tests for repo-search tool wiring."""

from __future__ import annotations

import json
from typing import cast

from juena.indexing.repo_manager import RepoManager
from juena.indexing.sparse_index import RepoSparseIndex
from juena.indexing.vector_index import RepoVectorIndex
from juena.tools import repo_search


def test_search_code_semantic_returns_compact_preview() -> None:
    content = "def login(user):\n" + ("x" * 400)

    class StubRepoManager:
        pass

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

    class StubRepoManager:
        pass

    class StubVectorIndex:
        pass

    class StubSparseIndex:
        pass

    class StubHybridHit:
        def __init__(self) -> None:
            self.file_path = "src/service.py"
            self.content = content
            self.is_doc = False
            self.sparse_rank = 1
            self.semantic_rank = 0
            self.rrf_score = 0.015873
            self.source = "both"

    original = repo_search.hybrid_search
    repo_search.hybrid_search = lambda *args, **kwargs: [StubHybridHit()]
    try:
        payload = json.loads(
            repo_search._search_code_hybrid(
                cast(RepoManager, StubRepoManager()),
                cast(RepoVectorIndex, StubVectorIndex()),
                cast(RepoSparseIndex, StubSparseIndex()),
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
            "sparse_rank": 1,
            "semantic_rank": 0,
            "rrf_score": 0.015873,
            "source": "both",
            "char_count": len(content),
            "preview": repo_search._compact_preview(content),
        }
    ]
    assert "content" not in payload[0]


def test_search_docs_local_returns_compact_preview() -> None:
    content = "README introduction " + ("z" * 260)

    class StubRepoManager:
        pass

    class StubVectorIndex:
        def search(self, repo_id: str, query: str, n_results: int = 10, where: dict | None = None) -> list[dict]:
            assert where == {"is_doc": True}
            return [
                {
                    "id": "doc-1",
                    "file_path": "README.md",
                    "chunk_index": 0,
                    "is_doc": True,
                    "distance": 0.03,
                    "content": content,
                }
            ]

    payload = json.loads(
        repo_search._search_docs_local(
            cast(RepoVectorIndex, StubVectorIndex()),
            "repo-a",
            "intro",
            max_results=1,
        )
    )

    assert payload == [
        {
            "id": "doc-1",
            "repo_id": "repo-a",
            "file_path": "README.md",
            "path": "/repos/repo-a/README.md",
            "chunk_index": 0,
            "is_doc": True,
            "distance": 0.03,
            "char_count": len(content),
            "preview": repo_search._compact_preview(content),
        }
    ]


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
