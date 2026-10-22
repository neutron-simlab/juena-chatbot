"""Tests for RepoManager: git clone, file listing, file reading."""

import os
import subprocess
from pathlib import Path

import pytest

from juena.indexing.repo_config import load_repo_configs
from juena.indexing.repo_manager import RepoManager


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


def test_unknown_repo(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    with pytest.raises(ValueError, match="Unknown repo"):
        mgr.resolve_root("no-such-repo")



def test_current_revision_returns_head_sha(repo_config_path: Path):
    mgr = _make_manager(repo_config_path)
    revision = mgr.current_revision("fake-repo")

    assert len(revision) == 40
    assert all(char in "0123456789abcdef" for char in revision)


def test_resolve_root_checks_out_the_revision_it_is_given(
    repo_config_path: Path, tmp_repo: Path
):
    """The commit juena-rag indexed, not the branch head that moved on since."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    indexed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_repo, check=True,
                             capture_output=True, text=True).stdout.strip()
    (tmp_repo / "src" / "later.py").write_text("LATER = True\n")
    subprocess.run(["git", "add", "."], cwd=tmp_repo, check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "later"], cwd=tmp_repo, check=True,
                   capture_output=True, env=env)
    mgr = _make_manager(repo_config_path)

    root = mgr.resolve_root("fake-repo", revision=indexed)

    assert mgr.current_revision("fake-repo") == indexed
    assert not (root / "src" / "later.py").exists()


def test_resolve_root_without_a_revision_follows_the_branch_head(
    repo_config_path: Path, tmp_repo: Path
):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_repo, check=True,
                          capture_output=True, text=True).stdout.strip()
    mgr = _make_manager(repo_config_path)

    mgr.resolve_root("fake-repo")

    assert mgr.current_revision("fake-repo") == head

