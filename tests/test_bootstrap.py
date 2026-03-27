"""Tests for repository bootstrap and readiness validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from juena.indexing import bootstrap
from juena.indexing.repo_config import RepoConfig, RepoSource
from juena.indexing.repo_manager import ManifestDiff


def _repo_config(repo_id: str) -> RepoConfig:
    return RepoConfig(
        id=repo_id,
        name=repo_id,
        source=RepoSource(url=f"https://example.com/{repo_id}.git"),
    )


class StubSparseIndex:
    def __init__(self, repo_manager: object) -> None:
        self.build_calls: list[str] = []

    def build_index(
        self,
        repo_id: str,
        chunks: list,
        *,
        force: bool = False,
        repo_revision: str | None = None,
    ) -> int:
        self.build_calls.append(repo_id)
        return len(chunks)

    def delete_file_chunks(self, repo_id: str, file_path: str) -> int:
        return 0


def test_bootstrap_repositories_rebuilds_stale_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staleness='repository revision changed' with no prior manifest triggers full rebuild."""
    configs = [_repo_config("repo-a"), _repo_config("repo-b")]
    created: dict[str, object] = {}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            assert configs_arg == configs
            self.repo_ids = [cfg.id for cfg in configs_arg]
            self.resolve_calls: list[str] = []
            created["repo_manager"] = self

        def resolve_root(self, repo_id: str) -> Path:
            self.resolve_calls.append(repo_id)
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return f"{repo_id}-rev"

        def build_manifest(self, repo_id: str) -> dict[str, str]:
            return {"file.py": "abc123"}

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager
            self.build_calls: list[tuple[str, bool]] = []
            created["vector_index"] = self

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            assert repo_revision == f"{repo_id}-rev"
            return "repository revision changed"

        def build_index(
            self,
            repo_id: str,
            *,
            force: bool = False,
            repo_revision: str | None = None,
            collect_chunks: bool = False,
        ) -> int | tuple[int, list]:
            self.build_calls.append((repo_id, force))
            assert repo_revision == f"{repo_id}-rev"
            count = {"repo-a": 3, "repo-b": 5}[repo_id]
            if collect_chunks:
                return count, [{"content": "x"}] * count
            return count

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "RepoSparseIndex", StubSparseIndex)
    monkeypatch.setattr(bootstrap, "_load_manifest", lambda repo_id: {})
    monkeypatch.setattr(bootstrap, "_save_manifest", lambda repo_id, manifest: None)

    results = bootstrap.bootstrap_repositories()

    repo_manager = created["repo_manager"]
    vector_index = created["vector_index"]
    assert results == {"repo-a": 3, "repo-b": 5}
    assert repo_manager.resolve_calls == ["repo-a", "repo-b"]
    assert vector_index.build_calls == [("repo-a", True), ("repo-b", True)]


def test_bootstrap_repositories_raises_immediately_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_repo_config("repo-a"), _repo_config("repo-b"), _repo_config("repo-c")]
    created: dict[str, object] = {}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]
            self.resolve_calls: list[str] = []
            created["repo_manager"] = self

        def resolve_root(self, repo_id: str) -> Path:
            self.resolve_calls.append(repo_id)
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return f"{repo_id}-rev"

        def build_manifest(self, repo_id: str) -> dict[str, str]:
            return {}

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.build_calls: list[tuple[str, bool]] = []
            created["vector_index"] = self

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "index configuration changed"

        def build_index(
            self,
            repo_id: str,
            *,
            force: bool = False,
            repo_revision: str | None = None,
            collect_chunks: bool = False,
        ) -> int | tuple[int, list]:
            self.build_calls.append((repo_id, force))
            assert repo_revision == f"{repo_id}-rev"
            if repo_id == "repo-b":
                raise RuntimeError("index failed")
            if collect_chunks:
                return 1, [{"content": "x"}]
            return 1

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "RepoSparseIndex", StubSparseIndex)
    monkeypatch.setattr(bootstrap, "_save_manifest", lambda repo_id, manifest: None)

    with pytest.raises(RuntimeError, match="index failed"):
        bootstrap.bootstrap_repositories()

    repo_manager = created["repo_manager"]
    vector_index = created["vector_index"]
    assert repo_manager.resolve_calls == ["repo-a", "repo-b"]
    assert vector_index.build_calls == [("repo-a", True), ("repo-b", True)]


def test_bootstrap_repositories_skips_rebuild_for_fresh_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_repo_config("repo-a")]
    seen: dict[str, bool] = {"built": False}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev"

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            assert repo_revision == "repo-a-rev"
            return None

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 7

        def build_index(
            self,
            repo_id: str,
            *,
            force: bool = False,
            repo_revision: str | None = None,
            collect_chunks: bool = False,
        ) -> int | tuple[int, list]:
            seen["built"] = True
            if collect_chunks:
                return 7, []
            return 7

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "RepoSparseIndex", StubSparseIndex)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 7}
    assert seen["built"] is False


def test_bootstrap_incremental_when_manifest_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When prior manifest exists and revision changed, incremental path is used."""
    configs = [_repo_config("repo-a")]
    created: dict[str, object] = {}
    incremental_called: list[str] = []
    saved_manifests: dict[str, dict] = {}

    old_manifest = {"a.py": "hash_old", "deleted.py": "hash_del"}
    new_manifest = {"a.py": "hash_new", "added.py": "hash_add"}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]
            created["repo_manager"] = self

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev-new"

        def build_manifest(self, repo_id: str) -> dict[str, str]:
            return new_manifest

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager
            created["vector_index"] = self

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "repository revision changed"

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 10

    def fake_incremental(rm, vi, si, repo_id, diff, rev):
        incremental_called.append(repo_id)
        assert isinstance(diff, ManifestDiff)
        assert "added.py" in diff.added
        assert "a.py" in diff.changed
        assert "deleted.py" in diff.deleted
        return 10

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "RepoSparseIndex", StubSparseIndex)
    monkeypatch.setattr(bootstrap, "_load_manifest", lambda repo_id: old_manifest)
    monkeypatch.setattr(bootstrap, "_save_manifest", lambda repo_id, m: saved_manifests.update({repo_id: m}))
    monkeypatch.setattr(bootstrap, "_incremental_reindex", fake_incremental)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 10}
    assert incremental_called == ["repo-a"]
    assert saved_manifests["repo-a"] == new_manifest


def test_validate_bootstrap_ready_requires_local_repos_and_indices() -> None:
    class StubRepoManager:
        repo_ids = ["repo-a", "repo-b", "repo-c"]

        def has_local_root(self, repo_id: str) -> bool:
            return repo_id != "repo-a"

    class StubVectorIndex:
        def collection_count_existing(self, repo_id: str) -> int | None:
            return {
                "repo-b": None,
                "repo-c": 0,
            }.get(repo_id, 4)

    with pytest.raises(RuntimeError, match="repo-a"):
        bootstrap.validate_bootstrap_ready(StubRepoManager(), StubVectorIndex())
