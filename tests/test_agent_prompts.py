from __future__ import annotations

import pytest

from juena.agents.specialist_runtime import PromptResourceError, load_markdown
from juena.agents.specialists.neutron_research.agent import build_neutron_research_prompt
from juena.agents.specialists.software.agent import build_software_prompt

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


def test_software_prompt_truthfully_describes_degraded_capabilities() -> None:
    prompt = build_software_prompt(has_repo_research=False, has_context7=False)

    assert "local repository index is unavailable" in prompt
    assert "Context7 is unavailable" in prompt
    assert "Do not claim to inspect local repositories" in prompt


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
