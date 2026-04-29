"""Tests for RepoManager: git clone, file listing, file reading."""

from pathlib import Path

import pytest

from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager, diff_manifests


def _make_manager(repo_config_path: Path) -> RepoManager:
    return RepoManager(load_repo_configs(repo_config_path))


def test_resolve_git_root(repo_config_path: Path, tmp_path: Path):
    mgr = _make_manager(repo_config_path)
    root = mgr.resolve_root("fake-repo")
    assert root.is_dir()
    assert (root / "src" / "main.py").exists()


def test_resolve_root_reclones_invalid_cache_dir(repo_config_path: Path, tmp_path: Path, monkeypatch):
    mgr = _make_manager(repo_config_path)
    dest = mgr.local_root("fake-repo")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "partial.txt").write_text("incomplete clone")

    clone_calls: list[tuple[str, Path, str]] = []
    pull_calls: list[tuple[Path, str]] = []

    def fake_clone(url: str, clone_dest: Path, branch: str) -> None:
        clone_calls.append((url, clone_dest, branch))
        clone_dest.mkdir(parents=True, exist_ok=True)
        (clone_dest / ".git").mkdir()

    def fake_pull(pull_dest: Path, branch: str) -> None:
        pull_calls.append((pull_dest, branch))

    monkeypatch.setattr(RepoManager, "_git_clone", staticmethod(fake_clone))
    monkeypatch.setattr(RepoManager, "_git_pull", staticmethod(fake_pull))

    root = mgr.resolve_root("fake-repo")

    assert root == dest
    assert clone_calls == [(mgr.get_config("fake-repo").source.url, dest, "main")]  # type: ignore[union-attr]
    assert pull_calls == []
    assert (dest / ".git").is_dir()
    assert not (dest / "partial.txt").exists()


def test_resolve_root_updates_origin_before_pull(repo_config_path: Path, monkeypatch):
    mgr = _make_manager(repo_config_path)
    dest = mgr.local_root("fake-repo")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / ".git").mkdir()

    set_origin_calls: list[tuple[Path, str]] = []
    pull_calls: list[tuple[Path, str]] = []

    def fake_set_origin(cache_dest: Path, url: str) -> None:
        set_origin_calls.append((cache_dest, url))

    def fake_pull(cache_dest: Path, branch: str) -> None:
        pull_calls.append((cache_dest, branch))

    monkeypatch.setattr(RepoManager, "_git_set_origin_url", staticmethod(fake_set_origin))
    monkeypatch.setattr(RepoManager, "_git_pull", staticmethod(fake_pull))

    root = mgr.resolve_root("fake-repo")

    assert root == dest
    assert set_origin_calls == [(dest, mgr.get_config("fake-repo").source.url)]  # type: ignore[union-attr]
    assert pull_calls == [(dest, "main")]


def test_list_files(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    files = mgr.list_files("fake-repo")
    assert "src/main.py" in files
    assert "README.md" in files
    assert "docs/guide.md" in files


def test_read_file(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    content = mgr.read_file("fake-repo", "src/main.py")
    assert "def greet" in content


def test_read_file_not_found(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    with pytest.raises(FileNotFoundError):
        mgr.read_file("fake-repo", "nonexistent.py")


def test_unknown_repo(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    with pytest.raises(ValueError, match="Unknown repo"):
        mgr.resolve_root("no-such-repo")


def test_list_repo_metadata(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    meta = mgr.list_repo_metadata()
    assert len(meta) == 1
    assert meta[0]["id"] == "fake-repo"
    assert meta[0]["source_url"].startswith("file://")


def test_current_revision_returns_head_sha(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    revision = mgr.current_revision("fake-repo")

    assert len(revision) == 40
    assert all(char in "0123456789abcdef" for char in revision)


def test_build_manifest(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    manifest = mgr.build_manifest("fake-repo")
    assert "src/main.py" in manifest
    assert len(manifest["src/main.py"]) == 16


def test_diff_manifests_detects_changes():
    old = {"a.py": "aaa", "b.py": "bbb", "deleted.py": "ddd"}
    new = {"a.py": "aaa", "b.py": "bbb_changed", "added.py": "eee"}
    diff = diff_manifests(old, new)
    assert diff.added == ["added.py"]
    assert diff.changed == ["b.py"]
    assert diff.deleted == ["deleted.py"]
    assert diff.has_changes


