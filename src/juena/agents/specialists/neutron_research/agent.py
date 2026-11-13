"""Build the first-class neutron-research specialist."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from juena.agents.personal_tools import PersonalToolsMiddleware
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
from juena_core.llms_providers import build_chat_model, same_model
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
# The software specialist's model and host. Qwen at JUWELS wrote only 8
# tokens/s at a 367k-token context; HZDR serves DeepSeek faster but only about
# 30 requests a minute, so juena_core paces it and JUWELS serves the same model
# when HZDR still refuses.
NEUTRON_RESEARCH_SPECIALIST_MODEL = BlabladorModelName.DEEPSEEK_V41_FLASH_HZDR.value
NEUTRON_RESEARCH_SPECIALIST_BACKUP_MODEL = BlabladorModelName.DEEPSEEK_V41_FLASH_JUWELS.value


def build_neutron_research_prompt(
    *, has_tavily: bool | None, unattended: bool = False
) -> str:
    """Append truthful runtime capability status to the authored prompt.

    Args:
        has_tavily: Whether web search is available; None leaves the status
            out, for when it is decided per user at run time.
        unattended: Whether this prompt is for the background-research copy.

    Returns:
        The system prompt.
    """

    prompt = load_markdown("juena.agents.specialists.neutron_research", "AGENT.md")
    status = (
        "Tavily web search is available for current facts and external evidence. "
        "Search returns excerpts; tavily_extract reads a selected page when more context is needed."
        if has_tavily
        else (
            "Tavily web search is unavailable. Distinguish established knowledge from "
            "claims that require current external verification and never invent citations."
        )
    )
    capabilities = prompt if has_tavily is None else f"{prompt}\n## Runtime capabilities\n\n- {status}\n"
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

    Args:
        tavily_tool: The web search tool, or None when it is not configured.
        summarizer_model: Model that summarizes a history that grows too long.
        fallback_models: Models to try when the main one fails.
        unattended: Build the background-research copy.
        checkpointer: Where the run saves its progress; background jobs pass
            their own.

    Returns:
        The specialist's name, description and compiled graph.
    """

    backend = build_specialist_backend()
    # Streaming and max_retries=0: see MODEL_MAX_RETRIES in juena_core.agents.specialist_runtime.
    model_settings = {
        "provider": NEUTRON_RESEARCH_SPECIALIST_PROVIDER,
        "temperature": 0.0,
        "max_tokens": 32_768,
        "streaming": True,
        "max_retries": 0,
    }
    model = build_chat_model(model=NEUTRON_RESEARCH_SPECIALIST_MODEL, **model_settings)
    backup = build_chat_model(model=NEUTRON_RESEARCH_SPECIALIST_BACKUP_MODEL, **model_settings)
    specialist_tools = list(build_tools(tavily_tool))
    if not unattended:
        specialist_tools.append(build_ask_user_tool(NEUTRON_RESEARCH_SPECIALIST_NAME))
    middleware = build_specialist_middleware(
        backend=backend,
        summarizer_model=summarizer_model,
        # A fallback that calls the same model would fail the same way.
        fallback_models=[m for m in (backup, *fallback_models) if not same_model(m, model)],
        filesystem_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
        specialist_name=NEUTRON_RESEARCH_SPECIALIST_NAME,
        unattended=unattended,
    )
    if tavily_tool is not None:
        middleware = [PersonalToolsMiddleware("tavily", ("tavily_search", "tavily_extract")), *middleware]
        if not unattended:
            middleware.extend([
                ToolCallLimitMiddleware(tool_name="tavily_search", run_limit=20),
                ToolCallLimitMiddleware(tool_name="tavily_extract", run_limit=6),
            ])
    runnable = create_agent(
        model=model,
        tools=specialist_tools,
        system_prompt=build_neutron_research_prompt(
            has_tavily=None if tavily_tool is not None else False,
            unattended=unattended,
        ),
        middleware=middleware,
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
