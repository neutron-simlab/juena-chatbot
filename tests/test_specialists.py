"""Tests for specialist-owned models, tools, backends, and skills."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.runnables import RunnableLambda

from juena_core.agents import specialist_runtime
from juena_core.agents.loop_guard import RepeatedToolCallMiddleware
from juena_core.agents.specialist_outcome import SpecialistOutcomeMiddleware
from juena.agents.specialists.neutron_research import agent as neutron_agent
from juena.agents.specialists.neutron_research import tools as neutron_tools
from juena.agents.specialists.software import agent as software_agent
from juena.agents.specialists.software import tools as software_tools
from juena_core.schema.llm_models import BlabladorModelName
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext


def test_software_tool_assembly_owns_repository_and_context7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_tools = [object(), object()]
    context7_tools = [object(), object()]
    repo_manager = object()
    vector_index = object()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        software_tools,
        "build_code_chat_tools",
        lambda manager, index: calls.append((manager, index)) or repository_tools,
    )

    assert software_tools.build_tools(
        repo_manager,
        vector_index,
        has_repo_research=True,
        context7_tools=context7_tools,
    ) == [*repository_tools, *context7_tools]
    assert calls == [(repo_manager, vector_index)]

    calls.clear()
    assert software_tools.build_tools(
        repo_manager,
        vector_index,
        has_repo_research=False,
        context7_tools=context7_tools,
    ) == context7_tools
    assert calls == []


def test_neutron_tool_assembly_adds_tavily_only_when_available() -> None:
    tavily = object()

    assert neutron_tools.build_tools(tavily) == [tavily]
    assert neutron_tools.build_tools(None) == []


def test_software_specialist_owns_fixed_model_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    model = object()
    backend = object()
    tools = [
        SimpleNamespace(name="search_code_semantic"),
        SimpleNamespace(name="resolve-library-id"),
    ]
    middleware = [object()]
    runnable = RunnableLambda(lambda state: state)
    repo_manager = SimpleNamespace(cache_dir=Path("/tmp/repos"))
    vector_index = object()

    monkeypatch.setattr(
        software_agent,
        "build_chat_model",
        lambda **kwargs: created.__setitem__("model_kwargs", kwargs) or model,
    )
    monkeypatch.setattr(
        software_agent,
        "build_specialist_backend",
        lambda **kwargs: created.__setitem__("backend_kwargs", kwargs) or backend,
    )
    monkeypatch.setattr(software_agent, "build_tools", lambda *_args, **_kwargs: tools)
    monkeypatch.setattr(
        software_agent,
        "build_specialist_middleware",
        lambda **kwargs: created.__setitem__("middleware_kwargs", kwargs) or middleware,
    )
    monkeypatch.setattr(
        software_agent,
        "create_agent",
        lambda **kwargs: created.__setitem__("agent_kwargs", kwargs) or runnable,
    )

    spec = software_agent.build_software_specialist(
        repo_manager=repo_manager,
        vector_index=vector_index,
        has_repo_research=True,
        context7_tools=[object()],
        summarizer_model=object(),
        fallback_models=[],
    )

    assert created["model_kwargs"] == {
        "provider": software_agent.SOFTWARE_SPECIALIST_PROVIDER,
        "model": software_agent.SOFTWARE_SPECIALIST_MODEL,
        "temperature": 0.0,
    }
    assert created["backend_kwargs"]["repo_cache_root"] == repo_manager.cache_dir
    assert created["agent_kwargs"]["model"] is model
    assert created["agent_kwargs"]["tools"][:-1] == list(tools)
    assert created["agent_kwargs"]["tools"][-1].name == "ask_user"
    assert created["agent_kwargs"]["middleware"] is middleware
    assert created["agent_kwargs"]["response_format"].schema.__name__ == "SpecialistReport"
    assert "protected_tool_names" not in created["middleware_kwargs"]
    assert created["agent_kwargs"]["context_schema"] is RuntimeModelContext
    assert spec["name"] == software_agent.SOFTWARE_SPECIALIST_NAME
    assert spec["description"] == software_agent.SOFTWARE_SPECIALIST_DESCRIPTION
    assert "Python or shell execution" in spec["description"]
    assert "with user approval" in spec["description"]
    assert spec["runnable"] is runnable


def test_software_specialist_defaults_to_minimax() -> None:
    assert (
        software_agent.SOFTWARE_SPECIALIST_MODEL
        == BlabladorModelName.MINIMAX_M27.value
    )


def test_neutron_specialist_owns_fixed_model_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}
    model = object()
    backend = object()
    tavily = object()
    tools = [tavily]
    middleware = [object()]
    runnable = RunnableLambda(lambda state: state)

    monkeypatch.setattr(
        neutron_agent,
        "build_chat_model",
        lambda **kwargs: created.__setitem__("model_kwargs", kwargs) or model,
    )
    monkeypatch.setattr(
        neutron_agent,
        "build_specialist_backend",
        lambda **kwargs: created.__setitem__("backend_kwargs", kwargs) or backend,
    )
    monkeypatch.setattr(neutron_agent, "build_tools", lambda _tool: tools)
    monkeypatch.setattr(
        neutron_agent,
        "build_specialist_middleware",
        lambda **kwargs: created.__setitem__("middleware_kwargs", kwargs) or middleware,
    )
    monkeypatch.setattr(
        neutron_agent,
        "create_agent",
        lambda **kwargs: created.__setitem__("agent_kwargs", kwargs) or runnable,
    )

    spec = neutron_agent.build_neutron_research_specialist(
        tavily_tool=tavily,
        summarizer_model=object(),
        fallback_models=[],
    )

    assert created["model_kwargs"] == {
        "provider": neutron_agent.NEUTRON_RESEARCH_SPECIALIST_PROVIDER,
        "model": neutron_agent.NEUTRON_RESEARCH_SPECIALIST_MODEL,
        "temperature": 0.0,
    }
    assert "repo_cache_root" not in created["backend_kwargs"]
    assert created["agent_kwargs"]["model"] is model
    assert created["agent_kwargs"]["tools"][:-1] == list(tools)
    assert created["agent_kwargs"]["tools"][-1].name == "ask_user"
    assert created["agent_kwargs"]["middleware"] is middleware
    assert created["agent_kwargs"]["response_format"].schema.__name__ == "SpecialistReport"
    assert created["agent_kwargs"]["context_schema"] is RuntimeModelContext
    assert spec["name"] == neutron_agent.NEUTRON_RESEARCH_SPECIALIST_NAME
    assert spec["description"] == neutron_agent.NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION
    assert spec["runnable"] is runnable


def _write_skill(skills_dir: Path, name: str = "plot-report") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Produce a specialist-owned artifact report.\n"
        "---\n\n"
        "# Plot report\n",
        encoding="utf-8",
    )
    return skill_dir


def test_skills_middleware_requires_a_valid_owned_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_skills = tmp_path / "empty"
    empty_skills.mkdir()
    invalid_skill = empty_skills / "invalid"
    invalid_skill.mkdir()
    (invalid_skill / "SKILL.md").write_text("# Missing frontmatter\n", encoding="utf-8")
    empty_body_skill = empty_skills / "empty-body"
    empty_body_skill.mkdir()
    (empty_body_skill / "SKILL.md").write_text(
        "---\nname: empty-body\ndescription: Has no authored instructions.\n---\n",
        encoding="utf-8",
    )
    owned_skills = tmp_path / "software-skills"
    _write_skill(owned_skills)

    monkeypatch.setattr(
        specialist_runtime,
        "create_summarization_middleware",
        lambda *_args: object(),
    )

    assert specialist_runtime.has_authored_skills(None) is False
    assert specialist_runtime.has_authored_skills(empty_skills) is False
    assert specialist_runtime.has_authored_skills(owned_skills) is True

    no_skills_backend = specialist_runtime.build_specialist_backend(skills_dir=empty_skills)
    owner_backend = specialist_runtime.build_specialist_backend(skills_dir=owned_skills)
    assert "/skills/" not in no_skills_backend.routes
    assert "/skills/" in owner_backend.routes

    no_skills_middleware = specialist_runtime.build_specialist_middleware(
        backend=no_skills_backend,
        summarizer_model=object(),
        fallback_models=[],
        filesystem_tool_descriptions={},
        skills_dir=empty_skills,
        skills_label="No skills",
        specialist_name="test-specialist",
    )
    owner_middleware = specialist_runtime.build_specialist_middleware(
        backend=owner_backend,
        summarizer_model=object(),
        fallback_models=[],
        filesystem_tool_descriptions={},
        skills_dir=owned_skills,
        skills_label="Software",
        specialist_name="test-specialist",
    )

    assert not any(isinstance(item, SkillsMiddleware) for item in no_skills_middleware)
    assert sum(isinstance(item, SkillsMiddleware) for item in owner_middleware) == 1
    assert sum(
        isinstance(item, SpecialistOutcomeMiddleware) for item in owner_middleware
    ) == 1
    assert not any(
        isinstance(item, TodoListMiddleware)
        for middleware in (no_skills_middleware, owner_middleware)
        for item in middleware
    )
    assert all(
        sum(isinstance(item, RepeatedToolCallMiddleware) for item in middleware) == 1
        for middleware in (no_skills_middleware, owner_middleware)
    )


def test_packaged_jscatter_skill_is_lazily_discoverable() -> None:
    skills_dir = software_agent._SPECIALIST_SKILLS_PATH
    backend = specialist_runtime.build_specialist_backend(skills_dir=skills_dir)
    middleware = SkillsMiddleware(
        backend=backend,
        sources=[("/skills/", "Software")],
    )

    update = middleware.before_agent({}, object(), {})

    assert update is not None
    metadata = update["skills_metadata"]
    jscatter = next(item for item in metadata if item["name"] == "jscatter")
    assert "neutron and X-ray scattering" in jscatter["description"]
    assert jscatter["path"] == "/skills/jscatter/SKILL.md"

    rendered = middleware._format_skills_list(metadata)
    assert "**jscatter**" in rendered
    assert "Read `/skills/jscatter/SKILL.md` for full instructions" in rendered
