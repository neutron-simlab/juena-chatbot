from __future__ import annotations

import pytest

from juena_core.agents.specialist_runtime import PromptResourceError, load_markdown
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
    assert "Never rerun specialist code" in prompt


def test_delegation_rules_live_where_they_are_read_rather_than_twice() -> None:
    """Rules stated in a tool description do not also belong in the system prompt.

    Both copies cost tokens on every model call; only the one attached to the
    tool is in front of the model at the moment it chooses. "Prefer one
    specialist" and "self-contained objective" were verbatim in both, and
    `subagent_type` is validated by the tool schema regardless of what either
    says.
    """
    from juena.agents.juena_agent import SUPERVISOR_TASK_DESCRIPTION

    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Prefer one specialist" in SUPERVISOR_TASK_DESCRIPTION
    assert "Prefer one specialist" not in prompt
    assert "self-contained objective" in SUPERVISOR_TASK_DESCRIPTION
    assert "Choose from the specialist names" not in prompt


def test_the_memory_contract_is_stated_only_in_the_memory_block() -> None:
    """`MemoryMiddleware` injects it on every call, so the prompt need not repeat it."""

    from juena_core.agents.backends import MEMORY_SYSTEM_PROMPT

    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Never paste the complete memory file" in MEMORY_SYSTEM_PROMPT
    assert "paste the complete memory file" not in prompt


def test_supervisor_prompt_makes_the_server_block_authoritative() -> None:
    """The supervisor writes the answer, so it must know which block wins."""

    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "`<specialist_report>`" in prompt
    assert "`<verified_by_server>`" in prompt
    assert "It is authoritative and overrides the report" in prompt
    assert "STATUS: UNVERIFIED" in prompt
    assert "Only present delivered outputs listed in the verified block" in prompt
    assert "their presence alone does not verify execution" in prompt


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents", "SUPERVISOR.md"),
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_prompts_tell_every_agent_when_to_ask_the_user(package: str, filename: str) -> None:
    """Each prompt names the tool and the cases specific to that agent."""

    prompt = load_markdown(package, filename)

    assert "`ask_user`" in prompt


def test_the_re_asking_rule_has_one_owner_and_covers_rewording() -> None:
    """The loop guard compares normalized arguments, so meaning is the prompt's job.

    A model that asks "which q-range?" and then "what q-range should I use?" is
    issuing two byte-different calls; nothing mechanical catches that. The rule
    lives on the tool all three agents share, rather than in three prompts.
    """
    from juena_core.agents.ask_user import ASK_USER_DESCRIPTION

    # Collapsed, so the assertion does not depend on where the text wraps.
    description = " ".join(ASK_USER_DESCRIPTION.split())

    assert "never reword an answered question and ask it again" in description
    assert "already answered" in description

    for package, filename in (
        ("juena.agents", "SUPERVISOR.md"),
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ):
        assert "ask the same question twice" not in load_markdown(package, filename)


def test_supervisor_routes_execution_without_promising_it() -> None:
    """`SUPERVISOR.md` is loaded verbatim, with no runtime capability block.

    Both specialists get one -- `build_software_prompt` states truthfully whether
    the sandbox is there. The supervisor cannot, so it must not claim execution
    is available: `SANDBOX_ENABLED` defaults to false, and the old wording
    ("Every authenticated user may request Python or shell execution") was simply
    untrue in that deployment. Route the request and trust what comes back.
    """
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "Delegate requests to run code" in prompt
    assert "`software-specialist`" in prompt
    assert "state that limitation accurately" in prompt
    assert "Every authenticated user may request" not in prompt
    assert "Never infer that execution is unavailable" not in prompt


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


def test_undelivered_files_are_reported_by_the_server_not_by_the_model() -> None:
    """The prompt asked for something the middleware does unconditionally.

    `SpecialistOutcomeMiddleware` appends every undelivered file to the report's
    `limitations` whether or not the specialist mentioned it, and lists them in
    `<verified_by_server>`. Instructing the model as well bought a duplicate that
    could be ignored.
    """
    import inspect

    from juena_core.agents import specialist_outcome

    source = inspect.getsource(specialist_outcome.SpecialistOutcomeMiddleware)
    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    assert "was not delivered" in source
    assert "limitations" in source
    assert "Produced but NOT delivered" in inspect.getsource(specialist_outcome._verified_block)
    assert "produced but not delivered" not in prompt


def test_software_prompt_explains_the_virtual_repos_route() -> None:
    """This one stays: it changes which tool the model picks, and nothing enforces it.

    A read-only backend refuses a *write* to `/repos`, but nothing stops the
    model passing `/repos/...` to `execute` as though it were a shell path, where
    it simply does not exist.
    """
    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    assert "/repos` is a virtual read-tool route" in prompt
    assert "never use it in `execute`" in prompt


def test_findings_keep_create_and_edit_apart_whatever_deep_agents_does() -> None:
    """The refusal is the backend's again, not a sentence in a prompt.

    Deep Agents 0.6.12 refused a `write_file` to an existing path with "...
    because it already exists" and sent the model to `edit_file`; 0.7 made the
    same call an upsert. The plan-02 cutover moved this application onto 0.7,
    and `/findings/<slug>.md` accumulates across turns -- so for a short while a
    second `write` silently replaced a finished report, with only the AGENT.md
    sentence below standing between the model and lost evidence.

    juena-core restored the contract where it belongs, in
    `FindingsStateBackend.write`, so prompt text is a courtesy again rather than
    the guard. Both halves are asserted here because the pair is the point: the
    prompt tells the model what to do, and the backend is what makes it true.
    """

    import inspect

    from deepagents.backends.state import StateBackend
    from juena_core.agents.backends import FindingsStateBackend

    # Upstream still upserts -- this test is not waiting for that to change.
    upstream = inspect.getsource(StateBackend.write)
    assert "because it already exists" not in upstream
    assert "update_file_data" in upstream

    guarded = inspect.getsource(FindingsStateBackend.write)
    assert "already exists" in guarded
    assert "edit_file" in guarded

    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")
    assert "extend it with `edit_file`" in prompt


def test_software_prompt_truthfully_describes_degraded_capabilities() -> None:
    prompt = build_software_prompt(has_repo_research=False, has_context7=False)

    assert "local repository index is unavailable" in prompt
    assert "Context7 is unavailable" in prompt
    assert "Do not claim to inspect local repositories" in prompt


def test_skill_loading_is_owned_by_the_middleware_that_injects_the_skills() -> None:
    """`SkillsMiddleware` supplies ~270 words of progressive-disclosure guidance.

    The authored prompt restated it in twelve lines of its own, so the model read
    the same procedure twice on every call, in two different phrasings. The
    middleware's copy is the one that also carries the actual skill listing.
    """
    from deepagents.middleware.skills import SKILLS_SYSTEM_PROMPT

    authored_prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")

    assert "SKILL.md" in SKILLS_SYSTEM_PROMPT
    assert "## Lazy skill loading" not in authored_prompt
    assert "no-skill decision" not in authored_prompt
    # Still true, and still worth pinning: the authored prompt must stay generic.
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


def test_the_software_prompt_names_no_search_tool_that_does_not_exist() -> None:
    """The prompt described `hybrid search` for a year after the tool was deleted.

    Inverted from the original: the prompt no longer has to introduce every tool,
    because each tool's own description does that at the call site. What it must
    never do is name one that is gone, which is the direction the incident ran.
    """
    import re

    from juena.tools.repo_search import build_code_chat_tools

    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")
    tool_names = {tool.name for tool in build_code_chat_tools(None, None)}
    mentioned = set(re.findall(r"`([a-z][a-z0-9_]{3,})`", prompt))
    claimed = {name for name in mentioned if name.startswith(("search_", "list_"))}

    assert claimed <= tool_names, f"prompt names tools that do not exist: {claimed - tool_names}"
    assert "hybrid" not in prompt.lower()


def test_supervisor_is_told_a_specialist_cannot_see_earlier_delegations() -> None:
    """It re-sent an unchanged objective to a specialist that had already failed.

    A subagent's state arrives with `messages` replaced by the objective alone, so
    nothing carries across invocations unless the supervisor writes it down.

    The claim is scoped to conversation text rather than to everything: the
    delegation boundary also passes `/inputs/` and `/findings/`, with shared
    workspace access, and an
    unqualified "only the objective" contradicted the mechanism it described.
    """
    prompt = load_markdown("juena.agents", "SUPERVISOR.md")

    assert "the only conversation text a specialist receives" in prompt
    assert "shared `/workspace` files, `/inputs/`, and `/findings/`" in prompt
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


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_writing_a_finding_is_triggered_by_the_objective_not_by_judgement(
    package: str, filename: str
) -> None:
    """Zero findings files had ever been written when this replaced the old wording.

    The first cause was mechanical -- `/findings/` resolved to the sandbox's
    read-only root, so every write raised `Errno 30`. The second was the prompt:
    it closed with "skip this only when you established nothing worth another
    specialist's time", a judgement call beside a report section that specifies
    every field. A specialist reads only the objective, so the trigger has to
    come from there.
    """
    prompt = load_markdown(package, filename)

    assert "Write one when the objective asks you to" in prompt
    assert "as binding as the report itself" in prompt
    assert "When the objective does not ask for one, do not write one" in prompt
    assert "Skip this only when" not in prompt


@pytest.mark.parametrize(
    ("package", "filename"),
    [
        ("juena.agents.specialists.software", "AGENT.md"),
        ("juena.agents.specialists.neutron_research", "AGENT.md"),
    ],
)
def test_the_finding_field_carries_the_answer_rather_than_a_summary_of_the_work(
    package: str, filename: str
) -> None:
    """"Put the shortest accurate result in `finding`" fed a re-delegation loop.

    For "list eight repositories and describe each", the shortest accurate
    result is "Eight repositories with their descriptions" -- which compresses
    the answer out of the report. The supervisor had nothing to reply with, so
    it delegated again, six times over, each report thinner than the last.
    """
    prompt = load_markdown(package, filename)

    assert "the answer, not a description of having found" in prompt
    assert "reports your activity, not your result" in prompt
    assert "Put the shortest accurate result" not in prompt


def test_the_supervisor_owns_whether_a_finding_gets_written() -> None:
    """The prompt has to match both the backend and the specialists' trigger.

    `ReadOnlyFindingsStateBackend` opens `/findings/` to the supervisor and keeps
    it unwritable. And because a specialist writes a finding only when the
    objective says so, the supervisor must know that an empty folder is the
    normal case -- an earlier version told it to run `ls /findings/` before
    answering, which returned `[]` and sent it back to delegate again.
    """
    # Collapsed, so an assertion does not depend on where the markdown wraps.
    prompt = " ".join(load_markdown("juena.agents", "SUPERVISOR.md").split())

    assert "when your objective asks it to" in prompt
    assert "never evidence that a delegation failed" in prompt
    assert "Re-delegate at most once on the same subject" in prompt
    assert "but never write to it" in prompt
    assert "reconcile them rather than delegating a third time" in prompt
    assert "you still cannot search repositories or run code" in prompt
