"""Build the first-class neutron-research specialist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from juena.agents.ask_user import build_ask_user_tool
from juena.agents.specialist_runtime import (
    build_specialist_backend,
    build_specialist_middleware,
    load_markdown,
)
from juena.agents.specialists.neutron_research.tools import (
    FILESYSTEM_TOOL_DESCRIPTIONS,
    build_tools,
)
from juena.core.llms_providers import build_chat_model
from juena.schema.agents import SpecialistReport
from juena.schema.llm_models import BlabladorModelName, Provider
from juena.server.agent.runtime_model_middleware import RuntimeModelContext

NEUTRON_RESEARCH_SPECIALIST_NAME = "neutron-research-specialist"
NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION = (
    "Research neutron science, instruments, methods, experiments, and broader "
    "factual questions within JueNA's scientific scope. Use this specialist for "
    "scientific explanations, comparisons, calculations, and current external evidence."
)
NEUTRON_RESEARCH_SPECIALIST_PROVIDER = Provider.BLABLADOR.value
NEUTRON_RESEARCH_SPECIALIST_MODEL = BlabladorModelName.GPT_OSS.value
_SPECIALIST_SKILLS_PATH = Path(__file__).with_name("skills")


def build_neutron_research_prompt(*, has_tavily: bool) -> str:
    """Append truthful runtime capability status to the authored prompt."""

    prompt = load_markdown("juena.agents.specialists.neutron_research", "AGENT.md")
    status = (
        "Tavily web search is available for current facts and external evidence. "
        "Each result carries the full page text alongside its short snippet."
        if has_tavily
        else (
            "Tavily web search is unavailable. Distinguish established knowledge from "
            "claims that require current external verification and never invent citations."
        )
    )
    return f"{prompt}\n## Runtime capabilities\n\n- {status}\n"


def build_neutron_research_specialist(
    *,
    tavily_tool: Any | None,
    summarizer_model: Any,
    fallback_models: list[Any],
) -> CompiledSubAgent:
    """Compile the neutron specialist with its own model, tools, and backend."""

    backend = build_specialist_backend(skills_dir=_SPECIALIST_SKILLS_PATH)
    model = build_chat_model(
        provider=NEUTRON_RESEARCH_SPECIALIST_PROVIDER,
        model=NEUTRON_RESEARCH_SPECIALIST_MODEL,
        temperature=0.0,
    )
    specialist_tools = [
        *build_tools(tavily_tool),
        build_ask_user_tool(NEUTRON_RESEARCH_SPECIALIST_NAME),
    ]
    runnable = create_agent(
        model=model,
        tools=specialist_tools,
        system_prompt=build_neutron_research_prompt(has_tavily=tavily_tool is not None),
        middleware=build_specialist_middleware(
            backend=backend,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
            filesystem_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
            skills_dir=_SPECIALIST_SKILLS_PATH,
            skills_label="Neutron research",
            specialist_name=NEUTRON_RESEARCH_SPECIALIST_NAME,
        ),
        response_format=ToolStrategy(SpecialistReport),
        context_schema=RuntimeModelContext,
        name="juena_neutron_research_specialist",
    )
    return {
        "name": NEUTRON_RESEARCH_SPECIALIST_NAME,
        "description": NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
        "runnable": runnable,
    }
