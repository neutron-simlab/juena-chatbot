from __future__ import annotations

import pytest

from juena.agents.specialist_runtime import PromptResourceError, load_markdown
from juena.agents.specialists.neutron_research.agent import build_neutron_research_prompt
from juena.agents.specialists.software.agent import build_software_prompt
from juena.sandbox.software import repository_software_requirements

REPORT_HEADINGS = (
    "## Finding",
    "## Evidence",
    "## Actions",
    "## Artifacts",
    "## Limitations",
)


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents", "SUPERVISOR.md"),
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_packaged_agent_prompts_load(package: str, filename: str) -> None:
    prompt = load_markdown(package, filename)
    assert prompt.strip()
    assert not prompt.startswith("---")


def test_missing_prompt_fails_loudly() -> None:
    with pytest.raises(PromptResourceError, match="Unable to load"):
        load_markdown("juena.agents", "MISSING.md")


def test_supervisor_prompt_enforces_strict_delegation_and_memory_scope() -> None:
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Delegate every substantive" in prompt
    assert "Use one specialist by default" in prompt
    assert "never paste the complete memory file" in prompt
    assert "Produce the only user-facing answer" in prompt
    assert "Never rerun specialist code" in prompt


def test_supervisor_routes_every_sandbox_request_to_the_software_specialist() -> None:
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Every authenticated user may request Python or shell execution" in prompt
    assert "Delegate requests to run code" in prompt
    assert "`software-specialist`" in prompt
    assert "Never infer that execution is unavailable" in prompt


@pytest.mark.parametrize(
    "prompt",
    [
        build_software_prompt(has_repo_research=True, has_context7=True),
        build_neutron_research_prompt(has_tavily=True),
    ],
)
def test_specialist_prompts_share_the_report_contract(prompt: str) -> None:
    assert all(heading in prompt for heading in REPORT_HEADINGS)
    assert "artifact ID | filename | MIME type | caption" in prompt


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents", "SUPERVISOR.md"),
        ("juena.agents.specialists.software", "AGENT.md"),
    ],
)
def test_prompts_forbid_linking_to_a_file_instead_of_attaching_it(
    package: str, filename: str
) -> None:
    """Artifacts are rendered by the application, so any link the model writes is dead.

    Left unsaid, a model whose file was refused reaches for the convention it
    knows from other sandboxes and emits `sandbox:/path/to/plot.png`.
    """
    prompt = load_markdown(package, filename)

    assert "sandbox:" in prompt
    assert "Markdown image" in prompt


def test_software_prompt_reports_files_that_were_not_delivered() -> None:
    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    assert "produced but not delivered" in prompt
    assert "Never claim the user can see a file the tool did not attach." in prompt


def test_software_prompt_truthfully_describes_degraded_capabilities() -> None:
    prompt = build_software_prompt(has_repo_research=False, has_context7=False)

    assert "local repository index is unavailable" in prompt
    assert "Context7 is unavailable" in prompt
    assert "Do not claim to inspect local repositories" in prompt


def test_software_prompt_requires_loading_applicable_dynamic_skills() -> None:
    authored_prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    skill_gate = authored_prompt.index("## Mandatory skill gate")
    repository_workflow = authored_prompt.index("For local repository questions")

    assert skill_gate < repository_workflow
    assert "before any repository search" in authored_prompt
    assert "This is mandatory, not a suggestion." in authored_prompt
    assert "dynamically injected **Available Skills** metadata" in authored_prompt
    assert "read its exact listed `SKILL.md`" in authored_prompt
    assert "load only the supporting references" in authored_prompt
    assert (
        "Repository documentation and remembered API knowledge do not replace"
        in authored_prompt
    )
    assert "Only when **Available Skills** is absent or empty" in authored_prompt
    assert "when no listed description" in authored_prompt
    assert "matches the objective" in authored_prompt
    assert "Do not load an unrelated" in authored_prompt
    assert "skill merely because one is available" in authored_prompt
    assert "jscatter" not in authored_prompt.lower()


def test_repository_software_manifest_starts_with_only_jscatter() -> None:
    assert repository_software_requirements() == ("jscatter==1.9.0.1",)


def test_software_prompt_names_preinstalled_repository_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "juena.agents.specialists.software.agent.sandbox_enabled", lambda: True
    )

    prompt = build_software_prompt(has_repo_research=True, has_context7=False)

    assert "`jscatter==1.9.0.1`" in prompt
    assert "runtime package installation remains forbidden" in prompt


def test_neutron_prompt_truthfully_describes_missing_web_search() -> None:
    prompt = build_neutron_research_prompt(has_tavily=False)

    assert "Tavily web search is unavailable" in prompt
    assert "never invent citations" in prompt


def test_neutron_prompt_requires_reading_raw_content_not_the_snippet() -> None:
    prompt = build_neutron_research_prompt(has_tavily=True)

    assert "Read each result's `raw_content`, not its `content` snippet" in prompt
    assert "never quote a value, definition, or table cell from it" in prompt


def test_neutron_prompt_requires_verifying_tabular_data() -> None:
    prompt = build_neutron_research_prompt(has_tavily=True)

    assert "never from position alone" in prompt
    assert "σ_coh = 4πb²" in prompt
    assert "are not file paths" in prompt
