"""Shared fixtures for code-chat-agent tests."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def tmp_repo(tmp_path: Path):
    """Create a tiny fake git repository and return its root."""
    repo = tmp_path / "origin"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    (src / "main.py").write_text(textwrap.dedent("""\
        def greet(name: str) -> str:
            \"\"\"Return a greeting message.\"\"\"
            return f"Hello, {name}!"

        if __name__ == "__main__":
            print(greet("world"))
    """))
    (src / "utils.py").write_text(textwrap.dedent("""\
        import os

        def get_env(key: str, default: str = "") -> str:
            return os.getenv(key, default)
    """))
    (repo / "README.md").write_text("# Fake Repo\n\nA tiny test repository.\n")
    docs = repo / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n\nHow to use the greet function.\n")

    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True,
                   capture_output=True, env=env)
    return repo


@pytest.fixture()
def repo_config_path(tmp_path: Path, tmp_repo: Path):
    """Write a repositories.yaml pointing at *tmp_repo* via file:// URL."""
    cfg = {
        "repositories": [
            {
                "id": "fake-repo",
                "name": "Fake Repo",
                "description": "A tiny test repository",
                "source": {
                    "url": tmp_repo.as_uri(),
                    "branch": "main",
                },
                "include_globs": ["**/*.py", "**/*.md"],
                "exclude_globs": ["**/.git/**"],
                "max_file_bytes": 524288,
                "chunk_size": 200,
                "chunk_overlap": 20,
                "docs_paths": ["README.md", "docs/"],
            }
        ]
    }
    p = tmp_path / "repositories.yaml"
    p.write_text(yaml.dump(cfg))

    os.environ["REPO_CACHE_DIR"] = str(tmp_path / "clones")
    yield p
    os.environ.pop("REPO_CACHE_DIR", None)
