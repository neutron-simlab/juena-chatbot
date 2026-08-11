"""Shared workspace ownership, path, and artifact tests."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from juena.sandbox.workspace import (
    DIRECTORY_MODE,
    WorkspaceOwnershipError,
    WorkspaceStore,
)


def test_path_resolution_rejects_traversal_symlinks_and_inputs_writes(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    inputs = tmp_path / "inputs"
    outside = tmp_path / "outside"
    workspace.mkdir()
    inputs.mkdir()
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="invalid_path"):
        WorkspaceStore.resolve_path(workspace, inputs, "/workspace/../secret", write=True)
    with pytest.raises(ValueError, match="invalid_path"):
        WorkspaceStore.resolve_path(
            workspace, inputs, "/workspace/escape/secret", write=False
        )
    with pytest.raises(PermissionError, match="permission_denied"):
        WorkspaceStore.resolve_path(workspace, inputs, "/inputs/data.csv", write=True)


def test_workspace_owner_cannot_change(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    store.ensure("a" * 64, "1" * 64)

    with pytest.raises(WorkspaceOwnershipError):
        store.ensure("a" * 64, "2" * 64)


def test_workspace_owner_marker_cannot_be_a_dangling_symlink(tmp_path) -> None:
    root = tmp_path / ("a" * 64)
    root.mkdir()
    (root / ".owner").symlink_to(tmp_path / "missing-owner")

    with pytest.raises(WorkspaceOwnershipError, match="owner marker"):
        WorkspaceStore(tmp_path).ensure("a" * 64, "1" * 64)


def test_collects_only_new_whitelisted_regular_files(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    store.ensure("a" * 64, "1" * 64)
    output_dir = store.ensure_outputs("a" * 64)
    (output_dir / "old.txt").write_text("old", encoding="utf-8")
    os.utime(output_dir / "old.txt", (1_600_000_000, 1_600_000_000))
    started_at = datetime.now(UTC)
    (output_dir / "new.py").write_text("print('ok')\n", encoding="utf-8")
    (output_dir / "blocked.html").write_text("<script>x</script>", encoding="utf-8")
    (output_dir / "escape.txt").symlink_to(Path("/etc/hosts"))

    scan = store.artifacts_since("a" * 64, started_at)

    assert scan.accepted == ("new.py",)
    assert scan.dropped == (("blocked.html", "Generated file type is not allowed"),)
    assert scan.omitted_drops == 0


def test_shared_directories_are_setgid_so_root_writes_stay_group_owned(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    store.ensure("a" * 64, "1" * 64)

    for path in (
        store.workspace_root("a" * 64),
        store.workspace_dir("a" * 64),
        store.inputs_dir("a" * 64),
        store.ensure_outputs("a" * 64),
    ):
        assert path.stat().st_mode & 0o7777 == DIRECTORY_MODE
