"""Contracts enforced by the development launcher itself."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_unit_gate_fails_closed_when_core_cannot_be_checked(tmp_path: Path) -> None:
    """A missing checkout must not turn the boundary check into a green run."""

    env = os.environ.copy()
    env["JUENA_CORE_REPO"] = str(tmp_path / "missing-core")

    result = subprocess.run(
        [str(ROOT / "juena"), "test", "-k", "this_test_must_never_start"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot check the import direction" in result.stderr
    assert "pytest" not in result.stdout
