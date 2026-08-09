"""Static packaging guards for authored agent Markdown resources."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_packages_the_complete_juena_source_tree() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/juena"
    ]
    assert (ROOT / "src/juena/agents/SUPERVISOR.md").is_file()
    assert sorted(ROOT.glob("src/juena/agents/specialists/*/AGENT.md"))


def test_container_context_keeps_prompts_and_future_specialist_skills() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY src/ ./src/" in dockerfile
    assert "COPY skills/ ./skills/" not in dockerfile
    assert "!src/**/SUPERVISOR.md" in dockerignore
    assert "!src/**/AGENT.md" in dockerignore
    assert "!src/**/skills/**/*.md" in dockerignore
