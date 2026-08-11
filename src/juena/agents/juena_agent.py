"""Assemble the JueNA supervisor and its explicitly registered specialists."""

from __future__ import annotations

from dataclasses import dataclass

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from juena.agents.ask_user import build_ask_user_tool
from juena.agents.backends import (
    JUENA_MEMORY_SYSTEM_PROMPT,
    MEMORY_SOURCES,
    SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS,
    build_supervisor_backend,
)
from juena.agents.loop_guard import RepeatedToolCallMiddleware
from juena.agents.specialist_runtime import (
    SUPERVISOR_MODEL_CALL_LIMIT,
    SUPERVISOR_TOOL_CALL_LIMIT,
    build_fallback_models,
    load_markdown,
    resilience_middleware,
)
from juena.agents.specialists.neutron_research import build_neutron_research_specialist
from juena.agents.specialists.software import build_software_specialist
from juena.core.llms_providers import build_chat_model
from juena.core.log import get_logger
from juena.indexing.bootstrap import describe_bootstrap_problems
from juena.indexing.repo_manager import RepoManager
from juena.indexing.vector_index import RepoVectorIndex
from juena.schema.llm_models import BlabladorModelName, Provider
from juena.server.agent.registry import register_agent_factory
from juena.server.agent.runtime_model_middleware import RuntimeModelContext, RuntimeModelMiddleware
from juena.sandbox.middleware import ArtifactMessageMiddleware
from juena.server.database.checkpointer import get_checkpointer
from juena.server.database.store import get_store
from juena.tools.context7 import Context7Runtime, load_optional_context7_tools
from juena.tools.tavily import load_optional_tavily_tool

logger = get_logger(__name__)

JUENA_AGENT_ID = "juena"
SUMMARIZER_PROVIDER = Provider.BLABLADOR.value
SUMMARIZER_MODEL = BlabladorModelName.GPT_OSS.value
SUPERVISOR_TASK_DESCRIPTION = """Delegate a substantive domain task to a specialist.
Give the specialist a self-contained objective with only the relevant user context and
saved preferences. Prefer one specialist; use more only for an explicit comparison or
clearly independent aspects that need different expertise. Resolve an ambiguous request
with ask_user before delegating rather than guessing on the user's behalf.

Every result arrives as a `<specialist_report>` block paired with a
`<verified_by_server>` block. The second block is written by the server from the actual
sandbox exit codes and the artifact store; it overrides anything the report claims.

Available specialist types:
{available_agents}
"""


@dataclass
class JuenaAgentResources:
    """Keep long-lived resources alive for the cached supervisor graph."""

    app: CompiledStateGraph
    context7_runtime: Context7Runtime | None = None


async def create_juena_agent(
    provider: str,
    model: str,
) -> tuple[JuenaAgentResources, CompiledStateGraph]:
    """Build the strict supervisor and both first-class specialists."""

    repo_manager = RepoManager()
    vector_index = RepoVectorIndex(repo_manager)
    bootstrap_problems = describe_bootstrap_problems(repo_manager, vector_index)
    has_repo_research = not bootstrap_problems
    if bootstrap_problems:
        logger.warning(
            "Local repository tools unavailable to the software specialist: %s. "
            "Run `python -m juena.indexing.bootstrap` to enable them.",
            "; ".join(bootstrap_problems),
        )

    supervisor_model = build_chat_model(provider=provider, model=model)
    summarizer_model = build_chat_model(
        provider=SUMMARIZER_PROVIDER,
        model=SUMMARIZER_MODEL,
        temperature=0.0,
    )
    context7_runtime = await load_optional_context7_tools()
    context7_tools = tuple(context7_runtime.tools) if context7_runtime is not None else ()
    tavily_tool = load_optional_tavily_tool()
    fallback_models = build_fallback_models()

    specialists = [
        build_software_specialist(
            repo_manager=repo_manager,
            vector_index=vector_index,
            has_repo_research=has_repo_research,
            context7_tools=context7_tools,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
        ),
        build_neutron_research_specialist(
            tavily_tool=tavily_tool,
            summarizer_model=summarizer_model,
            fallback_models=fallback_models,
        ),
    ]

    store = get_store()
    supervisor_backend = build_supervisor_backend(store)
    middleware = [
        # No TodoListMiddleware: the supervisor delegates, synthesizes, and asks.
        # A todo list adds no evidence and is the tool models most readily loop on.
        RepeatedToolCallMiddleware(),
        FilesystemMiddleware(
            backend=supervisor_backend,
            custom_tool_descriptions=SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS,
        ),
        MemoryMiddleware(
            backend=supervisor_backend,
            sources=MEMORY_SOURCES,
            system_prompt=JUENA_MEMORY_SYSTEM_PROMPT,
        ),
        # SUPERVISOR.md owns delegation policy. The task tool itself still carries
        # the authoritative specialist names and descriptions.
        SubAgentMiddleware(
            backend=supervisor_backend,
            subagents=specialists,
            system_prompt=None,
            task_description=SUPERVISOR_TASK_DESCRIPTION,
        ),
        create_summarization_middleware(summarizer_model, supervisor_backend),
        PatchToolCallsMiddleware(),
        ArtifactMessageMiddleware(),
        RuntimeModelMiddleware(),
        *resilience_middleware(
            fallback_models,
            model_call_limit=SUPERVISOR_MODEL_CALL_LIMIT,
            tool_call_limit=SUPERVISOR_TOOL_CALL_LIMIT,
        ),
    ]

    agent = create_agent(
        model=supervisor_model,
        tools=[build_ask_user_tool(JUENA_AGENT_ID)],
        system_prompt=load_markdown("juena.agents", "SUPERVISOR.md"),
        middleware=middleware,
        context_schema=RuntimeModelContext,
        checkpointer=get_checkpointer(),
        store=store,
        name=JUENA_AGENT_ID,
    ).with_config({"recursion_limit": 1000})

    resources = JuenaAgentResources(
        app=agent,
        context7_runtime=context7_runtime,
    )
    return (resources, agent)


register_agent_factory(JUENA_AGENT_ID, create_juena_agent, set_as_default=True)
