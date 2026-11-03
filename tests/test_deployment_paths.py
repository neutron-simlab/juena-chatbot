"""Installed wheels must use the deployment's external repository registry."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from juena.core import paths
from juena.deployment.doctor import repository_report
from juena.indexing import repo_config


def test_wheel_resolves_registry_from_explicit_app_root(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/juena/core/paths.py"
    package.parent.mkdir(parents=True)
    package.touch()
    deployment = tmp_path / "app"
    (deployment / "config").mkdir(parents=True)
    (deployment / "config/repositories.yaml").write_text(
        "repositories:\n  - id: jscatter\n    source:\n      url: https://example.org/repo.git\n"
    )
    monkeypatch.setattr(paths, "__file__", str(package))
    monkeypatch.setenv("JUENA_APP_ROOT", str(deployment))
    monkeypatch.delenv("REPO_CONFIG_PATH", raising=False)
    assert [repo.id for repo in repo_config.load_repo_configs()] == ["jscatter"]


def test_native_checkout_root_still_works(monkeypatch):
    monkeypatch.delenv("JUENA_APP_ROOT", raising=False)
    assert paths.repo_root() == Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("root", ["relative", "/missing-juena-app-root"])
def test_invalid_deployment_root_is_rejected(monkeypatch, root):
    monkeypatch.setenv("JUENA_APP_ROOT", root)
    with pytest.raises(RuntimeError, match="JUENA_APP_ROOT"):
        paths.repo_root()


def test_doctor_distinguishes_indexed_data_from_missing_clones():
    manager = SimpleNamespace(repo_ids=["jscatter"], has_local_root=lambda _: False)
    index = SimpleNamespace(ping=lambda: None, catalogue=lambda: [{"repo_id": "jscatter", "chunks": 1676}])
    report = repository_report(manager, index)
    assert report["reachable"] is True
    assert report["needs_ingestion"] is False
    assert report["cloned"] == []


def test_doctor_omits_sensitive_exception_messages():
    def broken():
        raise ConnectionError("https://user:secret@example.org")

    report = repository_report(SimpleNamespace(repo_ids=[]), SimpleNamespace(ping=broken))
    assert report["error_type"] == "ConnectionError"
    assert "secret" not in json.dumps(report)
