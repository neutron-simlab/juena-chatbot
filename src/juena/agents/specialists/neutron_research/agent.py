"""Build the first-class neutron-research specialist."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.base import BaseCheckpointSaver

from juena_core.agents.ask_user import build_ask_user_tool
from juena_core.agents.specialist_runtime import (
    UNATTENDED_NOTICE,
    build_specialist_backend,
    build_specialist_middleware,
    load_markdown,
)
from juena.agents.specialists.neutron_research.tools import (
    FILESYSTEM_TOOL_DESCRIPTIONS,
    build_tools,
)
from juena_core.llms_providers import build_chat_model
from juena_core.schema.agents import SpecialistReport
from juena_core.schema.llm_models import BlabladorModelName, Provider
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

NEUTRON_RESEARCH_SPECIALIST_NAME = "neutron-research-specialist"
NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION = (
    "Research neutron science, instruments, methods, experiments, and broader "
    "factual questions within JueNA's scientific scope. Use this specialist for "
    "scientific explanations, comparisons, calculations, and current external evidence. "
    "**It has no repository tools.** Questions about JueNA's indexed repositories, "
    "source code, or configuration go to the software specialist even when they are "
    "about neutron software -- sending them here produces a web search instead of an answer."
)
NEUTRON_RESEARCH_SPECIALIST_PROVIDER = Provider.BLABLADOR.value
NEUTRON_RESEARCH_SPECIALIST_MODEL = BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value


def build_neutron_research_prompt(
    *, has_tavily: bool, unattended: bool = False
) -> str:
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
    capabilities = f"{prompt}\n## Runtime capabilities\n\n- {status}\n"
    if not unattended:
        return capabilities
    return f"{capabilities}\n## Running unattended\n\n{UNATTENDED_NOTICE}\n"


def build_neutron_research_specialist(
    *,
    tavily_tool: Any | None,
    summarizer_model: Any,
    fallback_models: list[Any],
    unattended: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledSubAgent:
    """Compile the neutron specialist with its own model, tools, and backend.

    ``unattended`` compiles the copy that background research runs, without
    ``ask_user``: a job nobody is watching cannot have its question answered.
    """

    backend = build_specialist_backend()
    model = build_chat_model(
        provider=NEUTRON_RESEARCH_SPECIALIST_PROVIDER,
        model=NEUTRON_RESEARCH_SPECIALIST_MODEL,
        temperature=0.0,
        max_tokens=32_768,
    )
    specialist_tools = list(build_tools(tavily_tool))
    if not unattended:
        specialist_tools.append(build_ask_user_tool(NEUTRON_RESEARCH_SPECIALIST_NAME))
    runnable = create_agent(
        model=model,
        tools=specialist_tools,
        system_prompt=build_neutron_research_prompt(
            has_tavily=tavily_tool is not None,
            unattended=unattended,
        ),
        middleware=build_specialist_middleware(
            backend=backend,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
            filesystem_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
            specialist_name=NEUTRON_RESEARCH_SPECIALIST_NAME,
            unattended=unattended,
        ),
        response_format=ToolStrategy(SpecialistReport),
        context_schema=RuntimeModelContext,
        checkpointer=checkpointer,
        name="juena_neutron_research_specialist",
    )
    return {
        "name": NEUTRON_RESEARCH_SPECIALIST_NAME,
        "description": NEUTRON_RESEARCH_SPECIALIST_DESCRIPTION,
        "runnable": runnable,
    }
