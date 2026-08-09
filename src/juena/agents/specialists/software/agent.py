"""Build the first-class software specialist."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent

from juena.agents.specialist_runtime import (
    build_specialist_backend,
    build_specialist_middleware,
    load_markdown,
)
from juena.agents.specialists.software.tools import (
    FILESYSTEM_TOOL_DESCRIPTIONS,
    build_tools,
)
from juena.core.llms_providers import build_chat_model
from juena.schema.llm_models import BlabladorModelName, Provider
from juena.server.agent.runtime_model_middleware import RuntimeModelContext

SOFTWARE_SPECIALIST_NAME = "software-specialist"
SOFTWARE_SPECIALIST_DESCRIPTION = (
    "Investigate research software, staged code or errors, indexed JueNA "
    "repositories, and upstream library documentation. Use this specialist for "
    "code reading, repository behavior, configuration, debugging, and software guidance."
)
SOFTWARE_SPECIALIST_PROVIDER = Provider.BLABLADOR.value
SOFTWARE_SPECIALIST_MODEL = BlabladorModelName.MINIMAX_M27.value
_SPECIALIST_SKILLS_PATH = Path(__file__).with_name("skills")


def build_software_prompt(*, has_repo_research: bool, has_context7: bool) -> str:
    """Append truthful runtime capability status to the authored prompt."""

    prompt = load_markdown("juena.agents.specialists.software", "AGENT.md")
    repo_status = (
        "Indexed local repository search and read-only `/repos` access are available."
        if has_repo_research
        else (
            "The local repository index is unavailable. Do not claim to inspect local "
            "repositories or invent repository paths, code, or line numbers."
        )
    )
    context7_status = (
        "Context7 documentation tools are available for upstream libraries and frameworks."
        if has_context7
        else (
            "Context7 is unavailable. Do not claim to have checked upstream documentation; "
            "state when an answer requires external verification."
        )
    )
    return f"{prompt}\n## Runtime capabilities\n\n- {repo_status}\n- {context7_status}\n"


def build_software_specialist(
    *,
    repo_manager: Any,
    vector_index: Any,
    has_repo_research: bool,
    context7_tools: Sequence[Any],
    summarizer_model: Any,
    fallback_models: list[Any],
) -> CompiledSubAgent:
    """Compile the software specialist with its own model, tools, and backend."""

    backend = build_specialist_backend(
        repo_cache_root=repo_manager.cache_dir if has_repo_research else None,
        skills_dir=_SPECIALIST_SKILLS_PATH,
    )
    model = build_chat_model(
        provider=SOFTWARE_SPECIALIST_PROVIDER,
        model=SOFTWARE_SPECIALIST_MODEL,
        temperature=0.0,
    )
    runnable = create_agent(
        model=model,
        tools=build_tools(
            repo_manager,
            vector_index,
            has_repo_research=has_repo_research,
            context7_tools=context7_tools,
        ),
        system_prompt=build_software_prompt(
            has_repo_research=has_repo_research,
            has_context7=bool(context7_tools),
        ),
        middleware=build_specialist_middleware(
            backend=backend,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
            filesystem_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
            skills_dir=_SPECIALIST_SKILLS_PATH,
            skills_label="Software",
        ),
        context_schema=RuntimeModelContext,
        name="juena_software_specialist",
    )
    return {
        "name": SOFTWARE_SPECIALIST_NAME,
        "description": SOFTWARE_SPECIALIST_DESCRIPTION,
        "runnable": runnable,
    }
