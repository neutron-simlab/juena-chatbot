"""Tests for repository bootstrap and readiness validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from juena.retrieval import bootstrap
from juena.retrieval.repo_config import RepoConfig, RepoSource


def _repo_config(repo_id: str) -> RepoConfig:
    return RepoConfig(
        id=repo_id,
        name=repo_id,
        source=RepoSource(url=f"https://example.com/{repo_id}.git"),
    )


def test_bootstrap_repositories_rebuilds_stale_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        ) -> int:
            self.build_calls.append((repo_id, force))
            assert repo_revision == f"{repo_id}-rev"
            return {"repo-a": 3, "repo-b": 5}[repo_id]

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)

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
        ) -> int:
            self.build_calls.append((repo_id, force))
            assert repo_revision == f"{repo_id}-rev"
            if repo_id == "repo-b":
                raise RuntimeError("index failed")
            return 1

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)

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
        ) -> int:
            seen["built"] = True
            return 7

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 7}
    assert seen["built"] is False


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
