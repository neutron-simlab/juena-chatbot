"""Static packaging guards for authored agent Markdown resources."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
JSCATTER_SKILL = (
    ROOT / "src/juena/agents/specialists/software/skills/jscatter"
)


def test_wheel_packages_the_complete_juena_source_tree() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/juena"
    ]
    assert (ROOT / "src/juena/agents/SUPERVISOR.md").is_file()
    assert sorted(ROOT.glob("src/juena/agents/specialists/*/AGENT.md"))


def test_container_context_keeps_prompts_and_future_specialist_skills() -> None:
    """The checkout-only allowlist retains all authored source resources."""

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore = (ROOT / "Dockerfile.dockerignore").read_text(encoding="utf-8")
    patterns = [
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert "context: .\n" in compose

    # Source paths are relative to the chatbot checkout.
    assert "COPY src/ ./src/" in dockerfile
    assert "COPY juena-core/" not in dockerfile
    assert "COPY skills/ ./skills/" not in dockerfile

    # Admit build inputs; exclude credentials.
    assert patterns[0] == "*"
    assert "!src" in patterns
    assert "**/.env" in patterns
    assert "**/*.key" in patterns
    assert "*.md" not in patterns


def test_jscatter_skill_resources_and_interface_metadata_are_complete() -> None:
    skill_text = (JSCATTER_SKILL / "SKILL.md").read_text(encoding="utf-8")
    # Absolute paths, because models resolved relative links against /repos.
    reference_paths = set(re.findall(r"`/skills/jscatter/(references/[^`]+)`", skill_text))

    assert reference_paths == {
        "references/data-and-fitting.md",
        "references/dynamics-bio-and-dls.md",
        "references/example-data.md",
        "references/execution-recipes.md",
        "references/sas-and-static-models.md",
    }
    assert all((JSCATTER_SKILL / path).is_file() for path in reference_paths)
    assert "[TODO" not in skill_text
    assert re.search(r"They do not exist inside the\s+execution container", skill_text)
    assert "before any fit or" in skill_text
    assert "do not retry `write_file` with" in skill_text

    recipes = (
        JSCATTER_SKILL / "references/execution-recipes.md"
    ).read_text(encoding="utf-8")
    assert "is not mounted inside the" in recipes
    assert "before every fitting or plotting" in recipes
    assert "`write_file` will not overwrite" in recipes

    # Bundled example data is a real fallback, so the reference has to keep the
    # loader details that a plain js.dL gets wrong, and the honesty rule.
    example_data = (
        JSCATTER_SKILL / "references/example-data.md"
    ).read_text(encoding="utf-8")
    assert "js.examples.datapath" in example_data
    assert "Never present example data as the user's data." in example_data
    assert "There is no\n`js.readinx`" in example_data
    assert "unsupported locale setting" in example_data

    interface = yaml.safe_load(
        (JSCATTER_SKILL / "agents/openai.yaml").read_text(encoding="utf-8")
    )["interface"]
    assert interface == {
        "display_name": "JScatter",
        "short_description": "Analyze scattering experiments with JScatter",
        "default_prompt": (
            "Use $jscatter to analyze this neutron or X-ray scattering problem "
            "and produce a validated, reproducible result."
        ),
    }


def test_container_entrypoint_forwards_shutdown_to_both_servers() -> None:
    entrypoint = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")

    assert "FASTAPI_PID=$!" in entrypoint
    assert 'kill "$FASTAPI_PID"' in entrypoint
    assert 'kill "$STREAMLIT_PID"' in entrypoint
    assert 'wait "$FASTAPI_PID"' in entrypoint
    assert 'wait "$STREAMLIT_PID"' in entrypoint


def test_gateway_routes_every_application_api_path():
    """New API endpoints must not silently fall through to Streamlit."""
    from juena.server.service import app

    nginx = (ROOT / "deploy/nginx.conf").read_text()
    patterns = [re.compile(pattern) for pattern in re.findall(r"location ~ (\S+) \{", nginx)]
    ui_prefixes = re.findall(r"location \^~ (\S+) \{\s+proxy_pass http://\$backend:9501;", nginx)

    def routes_to_api(path):
        return not any(path.startswith(prefix) for prefix in ui_prefixes) and any(
            pattern.search(path) for pattern in patterns
        )

    for path in app.openapi()["paths"]:
        assert routes_to_api(path), path
    for path in ("/", "/_stcore/stream", "/static/app.js"):
        assert not routes_to_api(path), path


def test_sandbox_image_installs_repository_software_from_shared_manifest() -> None:
    dockerfile = (ROOT / "deploy/sandbox/Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY src/juena/sandbox/repository-requirements.txt "
        "/tmp/repository-requirements.txt"
    ) in dockerfile
    assert "--requirement /tmp/repository-requirements.txt" in dockerfile
    assert "import jscatter" in dockerfile
    assert re.search(r"apt-get install.*\bgrace\b", dockerfile, re.DOTALL)
    assert "shutil.which('gracebat')" in dockerfile
    assert "jscatter.GracePlot(headless=True)" in dockerfile
    assert "grace-smoke.png" in dockerfile
