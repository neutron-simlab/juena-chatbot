"""Build the first-class software specialist."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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
    has_authored_skills,
    load_markdown,
)
from juena.agents.specialists.software.tools import (
    FILESYSTEM_TOOL_DESCRIPTIONS,
    build_tools,
)
from juena_core.llms_providers import build_chat_model
from juena_core.schema.agents import SpecialistReport
from juena_core.schema.llm_models import BlabladorModelName, Provider
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena_core.sandbox.middleware import SandboxExecutionMiddleware
from juena_core.sandbox.runtime import build_sandbox_backend, sandbox_enabled
from juena.sandbox.software import repository_software_prompt

SOFTWARE_SPECIALIST_NAME = "software-specialist"
SOFTWARE_SPECIALIST_DESCRIPTION = (
    "Investigate research software, staged code or errors, indexed JueNA "
    "repositories, and upstream library documentation. Use this specialist for "
    "code reading, repository behavior, configuration, debugging, and software guidance. "
    "**It owns every question about JueNA's indexed repositories** -- which ones exist, "
    "what each is for, what is searchable -- including when the question also names "
    "neutron science, because it holds the only tools that can answer it. "
    "Delegate Python or shell execution, calculations, plots, and generated files here; "
    "when the sandbox is enabled, this specialist can execute them automatically in isolation."
)
SOFTWARE_SPECIALIST_PROVIDER = Provider.BLABLADOR.value
SOFTWARE_SPECIALIST_MODEL = BlabladorModelName.QWEN38_FLASH_NEXT_JUWELS.value
_SPECIALIST_SKILLS_PATH = Path(__file__).with_name("skills")


def build_software_prompt(
    *, has_repo_research: bool, has_context7: bool, unattended: bool = False
) -> str:
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
    execution_status = (
        "Sandboxed Python and shell execution is available. Write generated plots as PNG "
        "files under `/workspace/outputs`. Save reusable scripts and requested result files "
        "there too; the application collects, displays, and attaches them automatically. "
        f"{repository_software_prompt()} "
        "After a successful command reports collected artifacts, do not inspect, encode, "
        "or print those files. Never attempt package installation or network access."
        if sandbox_enabled() and not unattended
        else "Sandboxed execution is unavailable. Do not claim to run Python or generate plots."
    )
    capabilities = (
        f"{prompt}\n## Runtime capabilities\n\n"
        f"- {repo_status}\n- {context7_status}\n- {execution_status}\n"
    )
    if not unattended:
        return capabilities
    return f"{capabilities}\n## Running unattended\n\n{UNATTENDED_NOTICE}\n"


def build_software_specialist(
    *,
    repo_manager: Any,
    vector_index: Any,
    has_repo_research: bool,
    context7_tools: Sequence[Any],
    summarizer_model: Any,
    fallback_models: list[Any],
    unattended: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledSubAgent:
    """Compile the software specialist with its own model, tools, and backend.

    ``unattended`` compiles the copy that background research runs: no
    ``ask_user`` and no sandbox, because a job nobody is watching cannot answer
    an interrupt. Foreground commands run automatically, with execution evidence
    still recorded by the sandbox middleware.
    """

    repo_cache_root = repo_manager.cache_dir if has_repo_research else None
    can_execute = sandbox_enabled() and not unattended
    backend = (
        build_sandbox_backend(
            repo_cache_root=repo_cache_root,
            skills_dir=_SPECIALIST_SKILLS_PATH,
            has_skills=has_authored_skills(_SPECIALIST_SKILLS_PATH),
        )
        if can_execute
        else build_specialist_backend(
            repo_cache_root=repo_cache_root,
            workspace_writable=not unattended,
            skills_dir=_SPECIALIST_SKILLS_PATH,
        )
    )
    model = build_chat_model(
        provider=SOFTWARE_SPECIALIST_PROVIDER,
        model=SOFTWARE_SPECIALIST_MODEL,
        temperature=0.0,
        max_tokens=32_768,
    )
    specialist_tools = list(
        build_tools(
            repo_manager,
            vector_index,
            has_repo_research=has_repo_research,
            context7_tools=context7_tools,
        )
    )
    if not unattended:
        specialist_tools.append(build_ask_user_tool(SOFTWARE_SPECIALIST_NAME))
    runnable = create_agent(
        model=model,
        tools=specialist_tools,
        system_prompt=build_software_prompt(
            has_repo_research=has_repo_research,
            has_context7=bool(context7_tools),
            unattended=unattended,
        ),
        middleware=build_specialist_middleware(
            backend=backend,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
            filesystem_tool_descriptions=FILESYSTEM_TOOL_DESCRIPTIONS,
            skills_dir=_SPECIALIST_SKILLS_PATH,
            skills_label="Software",
            specialist_name=SOFTWARE_SPECIALIST_NAME,
            # Keep execution evidence and an explicit automatic policy together.
            # Unattended jobs receive neither execution nor questions.
            execution_middleware=(SandboxExecutionMiddleware(),) if can_execute else (),
            interrupt_on={"execute": False} if can_execute else None,
            unattended=unattended,
        ),
        response_format=ToolStrategy(SpecialistReport),
        context_schema=RuntimeModelContext,
        checkpointer=checkpointer,
        name="juena_software_specialist",
    )
    return {
        "name": SOFTWARE_SPECIALIST_NAME,
        "description": SOFTWARE_SPECIALIST_DESCRIPTION,
        "runnable": runnable,
    }
