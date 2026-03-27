"""Tests for repo_config: YAML parsing, defaults, and error handling."""

from pathlib import Path

import pytest
import yaml

from juena.indexing.repo_config import RepoConfig, load_repo_configs


def test_load_single_repo(repo_config_path: Path):
    configs = load_repo_configs(repo_config_path)
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.id == "fake-repo"
    assert cfg.name == "Fake Repo"
    assert cfg.source.url.startswith("file://")
    assert "**/*.py" in cfg.include_globs
    assert cfg.chunk_size == 200


def test_missing_config_returns_empty(tmp_path: Path):
    configs = load_repo_configs(tmp_path / "nonexistent.yaml")
    assert configs == []


def test_defaults_applied(tmp_path: Path):
    """Minimal entry should get sensible defaults."""
    cfg_data = {
        "repositories": [{
            "id": "minimal",
            "source": {"url": "https://github.com/org/repo.git"},
        }]
    }
    p = tmp_path / "repos.yaml"
    p.write_text(yaml.dump(cfg_data))
    configs = load_repo_configs(p)
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "minimal"
    assert cfg.source.branch == "main"
    assert cfg.chunk_size == 1500
    assert cfg.chunk_overlap == 200
    assert cfg.max_file_bytes == 524_288
    assert "README.md" in cfg.docs_paths


def test_missing_url_skipped(tmp_path: Path):
    """Entry without source.url should be skipped."""
    cfg_data = {"repositories": [{"id": "bad", "source": {}}]}
    p = tmp_path / "repos.yaml"
    p.write_text(yaml.dump(cfg_data))
    configs = load_repo_configs(p)
    assert configs == []
