"""Tests for vector index: build + semantic search."""

import os
from pathlib import Path
import subprocess

import pytest
from chromadb.api.types import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from juena.retrieval import vector_index as vector_index_module
from juena.retrieval.repo_config import load_repo_configs
from juena.retrieval.repo_manager import RepoManager
from juena.retrieval.vector_index import RepoVectorIndex


@pytest.fixture(autouse=True)
def _stub_embedding_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubConfig:
        BLABLADOR_API_KEY = None
        BLABLADOR_BASE_URL = None
        BLABLADOR_EMBEDDING_MODEL = "alias-qwen3-8b-embeddings"

    monkeypatch.setattr(vector_index_module, "_get_config", lambda: StubConfig)


def _make_index(repo_config_path: Path) -> tuple[RepoManager, RepoVectorIndex]:
    mgr = RepoManager(load_repo_configs(repo_config_path))
    vi = RepoVectorIndex(mgr)
    return mgr, vi


def test_build_and_count(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    count = vi.build_index("fake-repo", force=True)
    assert count > 0
    assert vi.collection_count_existing("fake-repo") == count


def test_semantic_search(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    vi.build_index("fake-repo", force=True)
    hits = vi.search("fake-repo", "greeting function", n_results=3)
    assert len(hits) > 0
    assert all("file_path" in h for h in hits)
    assert all("content" in h for h in hits)


def test_docs_only_filter(repo_config_path: Path):
    _, vi = _make_index(repo_config_path)
    vi.build_index("fake-repo", force=True)
    hits = vi.search("fake-repo", "guide", n_results=5, where={"is_doc": True})
    assert all(h["is_doc"] for h in hits)


def test_index_staleness_reason_is_clear_for_matching_revision(repo_config_path: Path) -> None:
    mgr, vi = _make_index(repo_config_path)
    revision = mgr.current_revision("fake-repo")

    vi.build_index("fake-repo", force=True, repo_revision=revision)

    assert vi.index_staleness_reason("fake-repo", revision) is None


def test_index_staleness_reason_detects_revision_changes(
    repo_config_path: Path,
    tmp_repo: Path,
) -> None:
    mgr, vi = _make_index(repo_config_path)
    initial_revision = mgr.current_revision("fake-repo")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    vi.build_index("fake-repo", force=True, repo_revision=initial_revision)

    (tmp_repo / "README.md").write_text("# Fake Repo\n\nUpdated contents.\n")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_repo, check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "update readme"], cwd=tmp_repo, check=True, capture_output=True, env=env)

    updated_root = mgr.resolve_root("fake-repo")
    subprocess.run(["git", "-C", str(updated_root), "fetch", "--depth", "1", "origin", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(updated_root), "reset", "--hard", "origin/main"], check=True, capture_output=True)
    updated_revision = mgr.current_revision("fake-repo")

    assert updated_revision != initial_revision
    assert vi.index_staleness_reason("fake-repo", updated_revision) == "repository revision changed"


def test_build_index_logs_percentage_progress(
    repo_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, vi = _make_index(repo_config_path)
    messages: list[str] = []
    original_info = vector_index_module.logger.info

    def capture_info(message: str, *args: object, **kwargs: object) -> None:
        rendered = message % args if args else message
        messages.append(rendered)
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(vector_index_module.logger, "info", capture_info)

    vi.build_index("fake-repo", force=True)

    progress_messages = [
        message for message in messages
        if message.startswith("Indexing progress for repo fake-repo:")
    ]
    assert progress_messages
    assert progress_messages[0].startswith("Indexing progress for repo fake-repo: 0% (0/")
    assert progress_messages[-1].startswith("Indexing progress for repo fake-repo: 100% (")


def test_build_embedding_function_uses_blablador_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubConfig:
        BLABLADOR_API_KEY = "test-key"
        BLABLADOR_BASE_URL = "https://blablador.example/v1"
        BLABLADOR_EMBEDDING_MODEL = "alias-qwen3-8b-embeddings"

    monkeypatch.setattr(vector_index_module, "_get_config", lambda: StubConfig)

    embedding_function = vector_index_module._build_embedding_function()

    assert isinstance(embedding_function, OpenAIEmbeddingFunction)
    assert embedding_function.model_name == "alias-qwen3-8b-embeddings"


def test_build_embedding_function_falls_back_without_blablador_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubConfig:
        BLABLADOR_API_KEY = None
        BLABLADOR_BASE_URL = None
        BLABLADOR_EMBEDDING_MODEL = "alias-qwen3-8b-embeddings"

    monkeypatch.setattr(vector_index_module, "_get_config", lambda: StubConfig)

    embedding_function = vector_index_module._build_embedding_function()

    assert isinstance(embedding_function, DefaultEmbeddingFunction)
