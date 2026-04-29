"""Tests for repository bootstrap and readiness validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from juena.indexing import bootstrap
from juena.indexing.repo_config import RepoConfig, RepoSource
from juena.indexing.repo_manager import ManifestDiff, RevisionDiff


def _repo_config(repo_id: str) -> RepoConfig:
    return RepoConfig(
        id=repo_id,
        name=repo_id,
        source=RepoSource(url=f"https://example.com/{repo_id}.git"),
    )


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

        def indexed_revision(self, repo_id: str) -> str | None:
            return None

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

        def indexed_revision(self, repo_id: str) -> str | None:
            return None

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

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 7}
    assert seen["built"] is False


def test_bootstrap_incremental_when_manifest_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When prior manifest exists and revision changed, incremental path is used."""
    configs = [_repo_config("repo-a")]
    incremental_called: list[str] = []
    saved_manifests: dict[str, dict] = {}

    old_manifest = {"a.py": "hash_old", "deleted.py": "hash_del"}
    new_manifest = {"a.py": "hash_new", "added.py": "hash_add"}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev-new"

        def build_manifest(self, repo_id: str) -> dict[str, str]:
            return new_manifest

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "repository revision changed"

        def indexed_revision(self, repo_id: str) -> str | None:
            return None

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 10

        def update_index_metadata(self, repo_id: str, repo_revision: str) -> None:
            return None

    def fake_incremental(rm, vi, repo_id, diff, rev):
        incremental_called.append(repo_id)
        assert isinstance(diff, ManifestDiff)
        assert "added.py" in diff.added
        assert "a.py" in diff.changed
        assert "deleted.py" in diff.deleted
        return 10

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
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


def test_incremental_reindex_updates_vector_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_incremental_reindex should update vector index for changed files."""

    class StubRepoManager:
        pass

    class StubVectorIndex:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []
            self.upserted: list[tuple[str, int]] = []

        def delete_file_chunks(self, repo_id: str, file_path: str) -> int:
            self.deleted.append((repo_id, file_path))
            return 0

        def build_file_chunks(self, repo_id: str, rel_path: str) -> list[dict[str, str | int | bool]]:
            return [
                {
                    "repo_id": repo_id,
                    "file_path": rel_path,
                    "chunk_index": 0,
                    "is_doc": False,
                    "content_hash": f"hash_{rel_path}_0",
                    "content": f"content of {rel_path} chunk 0",
                }
            ]

        def upsert_file_chunks(self, repo_id: str, chunks: list) -> int:
            self.upserted.append((repo_id, len(chunks)))
            return len(chunks)

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 42

    vector = StubVectorIndex()
    diff = ManifestDiff(added=["new.py"], changed=["changed.py"], deleted=["gone.py"])

    count = bootstrap._incremental_reindex(
        StubRepoManager(), vector,
        "repo-a", diff, "rev-123",
    )

    assert count == 42
    deleted_paths = {fp for _, fp in vector.deleted}
    assert "gone.py" in deleted_paths
    assert "changed.py" in deleted_paths
    assert len(vector.upserted) == 2


def test_bootstrap_revision_diff_uses_git_diff_for_incremental_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_repo_config("repo-a")]
    saved_manifests: dict[str, dict[str, str]] = {}
    incremental_diffs: list[ManifestDiff] = []
    metadata_updates: list[tuple[str, str]] = []

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev-new"

        def diff_revision_files(self, repo_id: str, old_revision: str, new_revision: str) -> RevisionDiff:
            assert old_revision == "repo-a-rev-old"
            assert new_revision == "repo-a-rev-new"
            return RevisionDiff(
                added=[],
                changed=["changed.py"],
                deleted=[],
                renamed=[("old_name.py", "new_name.py")],
            )

        def update_manifest_for_revision_diff(
            self,
            repo_id: str,
            manifest: dict[str, str],
            diff: RevisionDiff,
        ) -> dict[str, str]:
            assert manifest == {"changed.py": "old", "old_name.py": "rename-old"}
            assert diff.renamed == [("old_name.py", "new_name.py")]
            return {"changed.py": "new", "new_name.py": "rename-new"}

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.build_calls: list[tuple[str, bool]] = []

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "repository revision changed"

        def indexed_revision(self, repo_id: str) -> str | None:
            return "repo-a-rev-old"

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 12

        def update_index_metadata(self, repo_id: str, repo_revision: str) -> None:
            metadata_updates.append((repo_id, repo_revision))

        def build_index(
            self,
            repo_id: str,
            *,
            force: bool = False,
            repo_revision: str | None = None,
            collect_chunks: bool = False,
        ) -> int | tuple[int, list]:
            self.build_calls.append((repo_id, force))
            if collect_chunks:
                return 12, []
            return 12

    def fake_incremental(rm, vi, repo_id, diff, rev):
        incremental_diffs.append(diff)
        return 12

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(
        bootstrap,
        "_load_manifest",
        lambda repo_id: {"changed.py": "old", "old_name.py": "rename-old"},
    )
    monkeypatch.setattr(
        bootstrap,
        "_save_manifest",
        lambda repo_id, manifest: saved_manifests.update({repo_id: manifest}),
    )
    monkeypatch.setattr(bootstrap, "_incremental_reindex", fake_incremental)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 12}
    assert incremental_diffs == [
        ManifestDiff(
            added=["new_name.py"],
            changed=["changed.py"],
            deleted=["old_name.py"],
        )
    ]
    assert metadata_updates == [("repo-a", "repo-a-rev-new")]
    assert saved_manifests["repo-a"] == {"changed.py": "new", "new_name.py": "rename-new"}


def test_bootstrap_revision_diff_updates_metadata_only_when_indexed_files_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_repo_config("repo-a")]
    saved_manifests: dict[str, dict[str, str]] = {}
    metadata_updates: list[tuple[str, str]] = []
    incremental_called = {"called": False}

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev-new"

        def diff_revision_files(self, repo_id: str, old_revision: str, new_revision: str) -> RevisionDiff:
            return RevisionDiff(added=[], changed=[], deleted=[], renamed=[])

        def update_manifest_for_revision_diff(
            self,
            repo_id: str,
            manifest: dict[str, str],
            diff: RevisionDiff,
        ) -> dict[str, str]:
            assert not diff.has_changes
            return manifest

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "repository revision changed"

        def indexed_revision(self, repo_id: str) -> str | None:
            return "repo-a-rev-old"

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 7

        def update_index_metadata(self, repo_id: str, repo_revision: str) -> None:
            metadata_updates.append((repo_id, repo_revision))

    def fake_incremental(*args, **kwargs):
        incremental_called["called"] = True
        return 7

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "_load_manifest", lambda repo_id: {"src/main.py": "hash"})
    monkeypatch.setattr(
        bootstrap,
        "_save_manifest",
        lambda repo_id, manifest: saved_manifests.update({repo_id: manifest}),
    )
    monkeypatch.setattr(bootstrap, "_incremental_reindex", fake_incremental)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 7}
    assert incremental_called["called"] is False
    assert metadata_updates == [("repo-a", "repo-a-rev-new")]
    assert saved_manifests["repo-a"] == {"src/main.py": "hash"}


def test_bootstrap_falls_back_to_manifest_diff_when_git_diff_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [_repo_config("repo-a")]
    saved_manifests: dict[str, dict[str, str]] = {}
    metadata_updates: list[tuple[str, str]] = []
    incremental_diffs: list[ManifestDiff] = []

    class StubRepoManager:
        def __init__(self, configs_arg: list[RepoConfig]) -> None:
            self.repo_ids = [cfg.id for cfg in configs_arg]

        def resolve_root(self, repo_id: str) -> Path:
            return Path(f"/tmp/{repo_id}")

        def current_revision(self, repo_id: str) -> str:
            return "repo-a-rev-new"

        def diff_revision_files(self, repo_id: str, old_revision: str, new_revision: str) -> RevisionDiff:
            raise RuntimeError("missing base revision")

        def build_manifest(self, repo_id: str) -> dict[str, str]:
            return {"src/main.py": "new-hash", "src/extra.py": "extra-hash"}

    class StubVectorIndex:
        def __init__(self, repo_manager: StubRepoManager) -> None:
            self.repo_manager = repo_manager
            self.build_calls: list[tuple[str, bool]] = []

        def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
            return "repository revision changed"

        def indexed_revision(self, repo_id: str) -> str | None:
            return "repo-a-rev-old"

        def collection_count_existing(self, repo_id: str) -> int | None:
            return 11

        def update_index_metadata(self, repo_id: str, repo_revision: str) -> None:
            metadata_updates.append((repo_id, repo_revision))

        def build_index(
            self,
            repo_id: str,
            *,
            force: bool = False,
            repo_revision: str | None = None,
            collect_chunks: bool = False,
        ) -> int | tuple[int, list]:
            self.build_calls.append((repo_id, force))
            if collect_chunks:
                return 11, []
            return 11

    def fake_incremental(rm, vi, repo_id, diff, rev):
        incremental_diffs.append(diff)
        return 11

    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda config_path=None: configs)
    monkeypatch.setattr(bootstrap, "RepoManager", StubRepoManager)
    monkeypatch.setattr(bootstrap, "RepoVectorIndex", StubVectorIndex)
    monkeypatch.setattr(bootstrap, "_load_manifest", lambda repo_id: {"src/main.py": "old-hash"})
    monkeypatch.setattr(
        bootstrap,
        "_save_manifest",
        lambda repo_id, manifest: saved_manifests.update({repo_id: manifest}),
    )
    monkeypatch.setattr(bootstrap, "_incremental_reindex", fake_incremental)

    results = bootstrap.bootstrap_repositories()

    assert results == {"repo-a": 11}
    assert incremental_diffs == [
        ManifestDiff(
            added=["src/extra.py"],
            changed=["src/main.py"],
            deleted=[],
        )
    ]
    assert metadata_updates == [("repo-a", "repo-a-rev-new")]
    assert saved_manifests["repo-a"] == {"src/main.py": "new-hash", "src/extra.py": "extra-hash"}
