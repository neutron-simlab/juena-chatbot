from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from deepagents.middleware.skills import SkillsMiddleware
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from juena.agents.skills_backend import (
    JUENA_MEMORY_SYSTEM_PROMPT,
    MEMORY_SOURCES,
    ReadOnlyFilesystemBackend,
    ReadOnlyInputsStateBackend,
    build_user_context_backend,
    user_store_namespace,
)
from juena.server.agent.runtime_model_middleware import RuntimeModelContext

SKILL_BODY_MARKER = "Secret body marker"


def _write_skill(root: Path, name: str, description: str, body: str = SKILL_BODY_MARKER) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# Instructions\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_path


def _backend(root: Path) -> ReadOnlyFilesystemBackend:
    return ReadOnlyFilesystemBackend(root_dir=root, virtual_mode=True, label="read-only skill")


async def _discover_skills(root: Path) -> list[dict]:
    """Load skills the way the agent does, through the public middleware hook."""

    middleware = SkillsMiddleware(backend=_backend(root), sources=[("/", "Global")])
    update = await middleware.abefore_agent(
        {},
        SimpleNamespace(context=None, stream_writer=None, store=None),
        {},
    )
    return (update or {}).get("skills_metadata", [])


@pytest.mark.asyncio
async def test_startup_discloses_descriptions_but_not_skill_bodies(tmp_path: Path) -> None:
    _write_skill(tmp_path, "repo-review", "Review a repository safely.")

    skills = await _discover_skills(tmp_path)

    assert [(skill["name"], skill["description"]) for skill in skills] == [
        ("repo-review", "Review a repository safely.")
    ]
    # Progressive disclosure: the body only arrives when the agent reads the file.
    assert SKILL_BODY_MARKER not in str(skills)
    result = _backend(tmp_path).read(skills[0]["path"])
    assert result.file_data is not None
    assert SKILL_BODY_MARKER in result.file_data["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: missing-description",
        "name: [\ndescription: Broken",
    ],
)
async def test_malformed_skills_are_skipped(tmp_path: Path, frontmatter: str) -> None:
    skill_dir = tmp_path / "invalid"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nbody", encoding="utf-8")

    assert await _discover_skills(tmp_path) == []


@pytest.mark.asyncio
async def test_directories_without_a_skill_file_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "incomplete").mkdir()

    assert await _discover_skills(tmp_path) == []


def test_skills_backend_is_rooted_and_read_only(tmp_path: Path) -> None:
    _write_skill(tmp_path, "known", "Known skill")
    backend = _backend(tmp_path)

    with pytest.raises(ValueError, match="traversal"):
        backend.read("/../outside.md")
    assert backend.write("/known/SKILL.md", "replacement").error is not None
    assert backend.edit("/known/SKILL.md", "Known", "Changed").error is not None


def test_staged_inputs_are_read_only_for_every_agent() -> None:
    backend = ReadOnlyInputsStateBackend()

    assert backend.write("/inputs/uploads/report.csv", "tampered").error is not None
    assert backend.edit("/inputs/uploads/report.csv", "a", "b").error is not None


def test_memory_prompt_permits_writes_only_on_explicit_request() -> None:
    """Deep Agents' default prompt saves proactively; JueNA's must not."""

    from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT, MemoryMiddleware

    assert "might not explicitly ask you to remember" in MEMORY_SYSTEM_PROMPT
    assert "might not explicitly ask you to remember" not in JUENA_MEMORY_SYSTEM_PROMPT

    prompt = JUENA_MEMORY_SYSTEM_PROMPT.lower()
    assert "only when the user explicitly asks" in prompt
    assert "never infer" in prompt
    assert "never save credentials" in prompt
    assert "untrusted" in prompt

    # Accepted by the middleware it overrides (requires the {agent_memory} slot).
    middleware = MemoryMiddleware(
        backend=ReadOnlyInputsStateBackend(),
        sources=MEMORY_SOURCES,
        system_prompt=JUENA_MEMORY_SYSTEM_PROMPT,
    )
    assert middleware.sources == ["/memories/AGENTS.md"]


def test_memory_namespace_uses_trusted_user_uuid() -> None:
    alice = SimpleNamespace(context={"user_id": "00000000-0000-0000-0000-000000000001"})
    bob = SimpleNamespace(context={"user_id": "00000000-0000-0000-0000-000000000002"})

    assert user_store_namespace(alice) != user_store_namespace(bob)
    assert user_store_namespace(alice) == (
        "memories",
        "00000000-0000-0000-0000-000000000001",
    )

    with pytest.raises(ValueError, match="Authenticated user"):
        user_store_namespace(SimpleNamespace(context={}))


class _MemoryState(TypedDict):
    value: str


@pytest.mark.asyncio
async def test_memory_files_are_persisted_per_user(tmp_path: Path) -> None:
    store = InMemoryStore()
    backend = build_user_context_backend(str(tmp_path), store)

    async def write_agents_file(state: _MemoryState) -> dict:
        result = await backend.awrite("/memories/AGENTS.md", state["value"])
        assert result.error is None
        return {}

    builder = StateGraph(_MemoryState, context_schema=RuntimeModelContext)
    builder.add_node("write", write_agents_file)
    builder.add_edge(START, "write")
    builder.add_edge("write", END)
    graph = builder.compile(store=store)

    alice = "00000000-0000-0000-0000-000000000001"
    bob = "00000000-0000-0000-0000-000000000002"
    await graph.ainvoke(
        {"value": "# Preferences\n\n- Prefer Python examples.\n"},
        context=RuntimeModelContext(
            provider="openai",
            model="test-model",
            user_id=alice,
        ),
    )

    alice_file = await store.aget(("memories", alice), "/AGENTS.md")
    assert alice_file is not None
    assert "Prefer Python" in alice_file.value["content"]
    assert await store.aget(("memories", bob), "/AGENTS.md") is None
