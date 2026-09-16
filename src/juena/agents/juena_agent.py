"""Assemble the JueNA supervisor and its explicitly registered specialists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from juena_core.agents.ask_user import build_ask_user_tool
from juena_core.agents.backends import build_supervisor_backend
from juena_core.agents.specialist_outcome import ArtifactMessageMiddleware
from juena.research.delivery import ResearchDeliveryMiddleware
from juena_core.agents.specialist_runtime import (
    build_fallback_models,
    build_supervisor_middleware,
    load_markdown,
)
from juena.agents.specialists.neutron_research import build_neutron_research_specialist
from juena.agents.specialists.software import build_software_specialist
from juena_core.llms_providers import build_chat_model
from juena_core.log import get_logger
from juena.indexing.bootstrap import survey_repositories
from juena.indexing.repo_manager import RepoManager
from juena.indexing.rag_index import build_search_index
from juena_core.schema.llm_models import BlabladorModelName, Provider
from juena_core.server.agent.registry import register_agent_factory
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena_core.server.database.checkpointer import get_checkpointer
from juena_core.server.database.store import get_store
from juena.tools.context7 import Context7Runtime, load_optional_context7_tools
from juena.tools.research import build_research_tools
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
    #: The same specialists compiled for unattended work -- no `ask_user`, no
    #: sandbox. `juena.research.runner` invokes these directly; they are held
    #: here rather than inside `SubAgentMiddleware`, which owns only the
    #: interactive set and exposes them only through its `task` closure.
    background_specialists: dict[str, Any] = field(default_factory=dict)


async def create_juena_agent(
    provider: str,
    model: str,
) -> tuple[JuenaAgentResources, CompiledStateGraph]:
    """Build the strict supervisor and both first-class specialists."""

    repo_manager = RepoManager()
    vector_index = build_search_index()
    survey = survey_repositories(repo_manager, vector_index)
    bootstrap_problems = survey.problems
    # One unusable repository no longer removes the tools from the other eight,
    # and an unreachable store is reported per request rather than by deleting a
    # capability for the life of the process.
    has_repo_research = survey.any_searchable
    if bootstrap_problems:
        logger.warning(
            "Repository research is degraded (%d repository(s) usable): %s. "
            "A missing clone is fixed by `python -m juena.indexing.bootstrap "
            "--clone-only`; a missing index is the ingestion job's business.",
            len(survey.searchable),
            "; ".join(bootstrap_problems),
        )

    supervisor_model = build_chat_model(provider=provider, model=model)
    summarizer_model = build_chat_model(
        provider=SUMMARIZER_PROVIDER,
        model=SUMMARIZER_MODEL,
        temperature=0.0,
    )
    context7_runtime = await load_optional_context7_tools()
    context7_tools = (
        tuple(context7_runtime.tools) if context7_runtime is not None else ()
    )
    tavily_tool = load_optional_tavily_tool()
    fallback_models = build_fallback_models()

    def compile_specialists(*, unattended: bool) -> list:
        return [
            build_software_specialist(
                repo_manager=repo_manager,
                vector_index=vector_index,
                has_repo_research=has_repo_research,
                context7_tools=context7_tools,
                summarizer_model=summarizer_model,
                fallback_models=fallback_models,
                unattended=unattended,
            ),
            build_neutron_research_specialist(
                tavily_tool=tavily_tool,
                summarizer_model=summarizer_model,
                fallback_models=fallback_models,
                unattended=unattended,
            ),
        ]

    specialists = compile_specialists(unattended=False)
    # A second compile, not a second set of models: `build_chat_model` caches on
    # (provider, model, temperature), so this costs tool objects and a prompt
    # string. The two sets must stay one graph apart -- an unattended specialist
    # that could still reach `ask_user` would block forever on nobody.
    background_specialists = {
        spec["name"]: spec["runnable"] for spec in compile_specialists(unattended=True)
    }

    store = get_store()
    supervisor_backend = build_supervisor_backend(store)
    # The stack, its order and the no-TodoListMiddleware decision are core's --
    # middleware order is nesting order, and a core test asserts these class
    # names in sequence. `extra` is its one splice point, between
    # PatchToolCallsMiddleware and RuntimeModelMiddleware, which is exactly
    # where these two used to sit.
    middleware = build_supervisor_middleware(
        backend=supervisor_backend,
        summarizer_model=summarizer_model,
        fallback_models=fallback_models,
        # Raw specialists: `build_supervisor_middleware` applies the delegation
        # boundary itself. See `juena_core.agents.delegation` for what may cross
        # into a specialist and what comes back.
        subagents=specialists,
        # SUPERVISOR.md owns delegation policy. The task tool itself still carries
        # the authoritative specialist names and descriptions. Passed explicitly
        # because core's default says "execution evidence" where this one says
        # "sandbox exit codes", and prompt text does not change in this cutover.
        task_description=SUPERVISOR_TASK_DESCRIPTION,
        extra=(
            ArtifactMessageMiddleware(),
            # Declares the private, checkpointed marker `check_research` stamps
            # when it delivers a background result, which is what makes
            # collection idempotent across a crash on either side of the
            # checkpoint. Research is this application's, not core's.
            ResearchDeliveryMiddleware(),
        ),
    )

    agent = create_agent(
        model=supervisor_model,
        tools=[
            build_ask_user_tool(JUENA_AGENT_ID),
            *build_research_tools(list(background_specialists)),
        ],
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
        background_specialists=background_specialists,
    )
    return (resources, agent)


register_agent_factory(JUENA_AGENT_ID, create_juena_agent, set_as_default=True)
