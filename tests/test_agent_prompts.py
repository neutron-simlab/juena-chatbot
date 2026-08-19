from __future__ import annotations

import pytest

from juena.agents.specialist_runtime import PromptResourceError, load_markdown
from juena.agents.specialists.neutron_research.agent import build_neutron_research_prompt
from juena.agents.specialists.software.agent import build_software_prompt
from juena.sandbox.software import repository_software_requirements

STRUCTURED_REPORT_FIELDS = ("status", "finding", "evidence", "actions", "limitations")


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
    assert "Prefer one specialist" in prompt
    assert "never paste the complete memory file" in prompt
    assert "Never rerun specialist code" in prompt


def test_supervisor_prompt_makes_the_server_block_authoritative() -> None:
    """The supervisor writes the answer, so it must know which block wins."""

    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "`<specialist_report>`" in prompt
    assert "`<verified_by_server>`" in prompt
    assert "It is authoritative and overrides the report" in prompt
    assert "STATUS: UNVERIFIED" in prompt
    assert "Only mention files the verified block actually lists" in prompt


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents", "SUPERVISOR.md"),
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_prompts_tell_every_agent_when_to_ask_the_user(package: str, filename: str) -> None:
    prompt = load_markdown(package, filename)

    assert "`ask_user`" in prompt
    assert "The user is the domain expert; you are not." in prompt
    assert "never ask the same question twice" in prompt or "never re-ask" in prompt


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
    assert "## Structured report" in prompt
    assert all(f"`{field}`" in prompt for field in STRUCTURED_REPORT_FIELDS)


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


def test_software_prompt_explains_virtual_repos_and_safe_script_revision() -> None:
    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    assert "/repos` is a virtual read-tool route" in prompt
    assert "never use it in `execute`" in prompt
    assert "`write_file` creates a new path and refuses an existing one" in prompt
    assert "with `edit_file` or choose a new filename" in prompt


def test_software_prompt_truthfully_describes_degraded_capabilities() -> None:
    prompt = build_software_prompt(has_repo_research=False, has_context7=False)

    assert "local repository index is unavailable" in prompt
    assert "Context7 is unavailable" in prompt
    assert "Do not claim to inspect local repositories" in prompt


def test_software_prompt_preserves_model_chosen_lazy_skill_loading() -> None:
    authored_prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    skill_loading = authored_prompt.index("## Lazy skill loading")
    repository_workflow = authored_prompt.index("For local repository questions")

    assert skill_loading < repository_workflow
    assert "dynamically injected **Available Skills** metadata" in authored_prompt
    assert "Decide from each listed description" in authored_prompt
    assert "read its exact listed `SKILL.md`" in authored_prompt
    assert "load only the supporting references" in authored_prompt
    assert "If no listed skill is relevant, continue directly" in authored_prompt
    assert "Do not call or report a" in authored_prompt
    assert "no-skill decision" in authored_prompt
    assert "They do not block" in authored_prompt
    assert "skill_decision" not in authored_prompt
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


def test_software_prompt_only_names_search_tools_that_exist() -> None:
    """The prompt described `hybrid search` for a year after the tool was deleted.

    Derived from `build_code_chat_tools` rather than hard-coded, so deleting a
    tool fails here instead of leaving the model reaching for it.
    """
    from juena.tools.repo_search import build_code_chat_tools

    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")
    tool_names = {tool.name for tool in build_code_chat_tools(None, None)}

    for name in tool_names:
        assert name in prompt, f"{name} is a real tool the prompt never offers"
    assert "hybrid" not in prompt.lower()


def test_supervisor_is_told_a_specialist_cannot_see_earlier_delegations() -> None:
    """It re-sent an unchanged objective to a specialist that had already failed.

    A subagent's state arrives with `messages` replaced by the objective alone, so
    nothing carries across invocations unless the supervisor writes it down.
    """
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "only the `objective` you write" in prompt
    assert "any earlier delegation" in prompt
    assert "Delegating again on the same subject" in prompt
    assert "do not retry" in prompt


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_both_specialists_are_told_to_use_the_shared_findings_channel(
    package: str, filename: str
) -> None:
    """A subagent inherits `files` but never `messages`, so this is the only channel.

    Both sides have to know about it: one specialist writing where the other never
    looks accumulates nothing.
    """
    prompt = load_markdown(package, filename)

    assert "ls /findings/" in prompt
    assert "/findings/<short-topic-slug>.md" in prompt
    assert "extend it with `edit_file`" in prompt


def test_the_supervisor_forwards_findings_paths_rather_than_their_contents() -> None:
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Specialists leave their evidence in `/findings/`" in prompt
    assert "the file holds the paths and line numbers" in prompt
