"""Tests for RepoManager: git clone, file listing, file reading."""

from pathlib import Path

import pytest

from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import ManifestDiff, RepoManager, diff_manifests


def _make_manager(repo_config_path: Path) -> RepoManager:
    return RepoManager(load_repo_configs(repo_config_path))


def test_resolve_git_root(repo_config_path: Path, tmp_path: Path):
    mgr = _make_manager(repo_config_path)
    root = mgr.resolve_root("fake-repo")
    assert root.is_dir()
    assert (root / "src" / "main.py").exists()


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


def test_diff_manifests_no_changes():
    m = {"a.py": "aaa", "b.py": "bbb"}
    diff = diff_manifests(m, m)
    assert not diff.has_changes
    assert diff.added == []
    assert diff.changed == []
    assert diff.deleted == []
