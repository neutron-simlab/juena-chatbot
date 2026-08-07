"""Registry for the cached JüNA graph and its compatibility aliases."""
import inspect
from typing import Any, Callable, Awaitable
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.core.config import global_config
from juena.schema.llm_models import Provider, get_default_model_for_provider
from juena.server.errors import AgentNotFoundError

logger = get_logger(__name__)

# Provider/model are handled per invocation via runtime context. The factory's
# pair only supplies the initial defaults used while the graph is constructed.
#
# `register_agent_factory(set_as_default=True)` rebinds this at import time, so
# it must never be read into another module's namespace at *its* import time.
# Callers use `get_default_agent()` instead; a `from ... import DEFAULT_AGENT`
# captures whatever value happened to be current when that module was loaded.
DEFAULT_AGENT = "juena"

# Type alias for agent instance (can be any type)
AgentInstance = Any

# Registry storage: agent_id -> (AgentInstance, CompiledStateGraph)
_agent_registry: dict[str, tuple[AgentInstance, CompiledStateGraph]] = {}

# Agent factory registry: agent_id -> async factory function
# Factory function signature: async (provider: str, model: str) -> tuple[AgentInstance, CompiledStateGraph]
_agent_factories: dict[str, Callable[[str, str], Awaitable[tuple[AgentInstance, CompiledStateGraph]]]] = {}

# Retired agent ids that should still resolve: alias -> canonical agent_id
_agent_aliases: dict[str, str] = {}

def register_agent_factory(
    agent_id: str,
    factory: Callable[[str, str], Awaitable[tuple[AgentInstance, CompiledStateGraph]]],
    set_as_default: bool = False
) -> None:
    """
    Register a factory function for creating agents of a specific type.

    Args:
        agent_id: Agent identifier (e.g., "my_agent")
        factory: Async factory function that takes (provider, model) and returns (AgentInstance, CompiledStateGraph)
        set_as_default: If True, set this agent as the default agent
    """
    _agent_factories[agent_id] = factory
    logger.info(f"Registered factory for agent type: {agent_id}")

    if set_as_default:
        global DEFAULT_AGENT
        DEFAULT_AGENT = agent_id
        logger.info(f"Set {agent_id} as default agent")


def get_default_agent() -> str:
    """The agent id to use when a request does not name one.

    Read through this rather than importing ``DEFAULT_AGENT``: the value is
    rebound when an agent registers itself as the default, which happens after
    most modules have already been imported.
    """
    return DEFAULT_AGENT


def register_agent_alias(alias: str, agent_id: str) -> None:
    """
    Make a retired agent id resolve to a current one.

    Aliases are not listed by ``list_registered_agents``: they exist so that
    clients holding an old id keep working, not to advertise extra agents.

    Args:
        alias: The retired identifier (e.g., "react_agent")
        agent_id: The identifier it should resolve to
    """
    _agent_aliases[alias] = agent_id
    logger.info(f"Registered agent alias: {alias} -> {agent_id}")


def _resolve_agent_id(agent_id: str) -> str:
    """Map a possibly-retired agent id onto the agent that serves it."""
    resolved = _agent_aliases.get(agent_id)
    if resolved is None:
        return agent_id
    logger.debug(f"Resolved legacy agent id {agent_id} -> {resolved}")
    return resolved


def _normalize_provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    """
    Normalize provider and model to default values if not provided.
    
    Args:
        provider: LLM provider (optional)
        model: LLM model name (optional)
        
    Returns:
        Tuple of (provider, model)
    """
    # Determine provider (default to config default)
    if provider is None:
        provider = global_config.DEFAULT_PROVIDER
    
    # Determine model (default to provider's default)
    if model is None:
        try:
            provider_enum = Provider(provider)
            model = get_default_model_for_provider(provider_enum)
        except ValueError:
            # Invalid provider, fall back to config default
            provider = global_config.DEFAULT_PROVIDER
            model = global_config.DEFAULT_MODEL
    
    return provider, model


async def get_agent(
    agent_id: str, 
    provider: str | None = None, 
    model: str | None = None
) -> CompiledStateGraph:
    """
    Get an agent by ID, creating it if it doesn't exist.
    
    Provider and model are used only for initial creation (to set default LLMs).
    After creation, model selection is handled per invocation via runtime context.
    
    Args:
        agent_id: Agent identifier (e.g., "my_agent")
        provider: LLM provider (openai or blablador). Defaults to global_config.DEFAULT_PROVIDER.
                  Only used for initial creation; ignored for existing agents.
        model: LLM model name. Defaults to provider's default model.
               Only used for initial creation; ignored for existing agents.
    
    Returns:
        CompiledStateGraph for the agent
        
    Raises:
        AgentNotFoundError: If agent_id is not registered or creation fails
    """
    # Normalize provider/model for initial creation (if needed)
    provider, model = _normalize_provider_model(provider, model)
    agent_id = _resolve_agent_id(agent_id)

    logger.info(f"Getting agent: {agent_id} (provider={provider}, model={model} used only for initial creation)")

    if agent_id not in _agent_registry:
        # Check if factory is registered for this agent type
        if agent_id not in _agent_factories:
            logger.error(f"Unknown agent requested: {agent_id}")
            available_agents = list(_agent_factories.keys())
            raise AgentNotFoundError(
                agent_id,
                details={"available_agents": available_agents}
            )
        
        try:
            factory = _agent_factories[agent_id]
            logger.info(f"Creating new {agent_id} agent with default provider={provider}, model={model}")
            
            # Call factory function to create agent (provider/model used for initial LLM setup)
            agent_instance, compiled_graph = await factory(provider, model)
            
            _agent_registry[agent_id] = (agent_instance, compiled_graph)
            logger.info(f"{agent_id} agent created and registered")
        except Exception as e:
            logger.error(f"Failed to create {agent_id} agent: {e}")
            raise AgentNotFoundError(
                agent_id,
                details={"error": str(e), "agent_type": agent_id, "provider": provider, "model": model}
            )
    else:
        logger.info(f"Using existing agent: {agent_id}")
    
    # Return the CompiledStateGraph (app) from the registry
    return _agent_registry[agent_id][1]


async def shutdown_agents() -> None:
    """Release long-lived resources held by cached agents.

    Agent factories may hold network clients (the Context7 MCP session) that
    stay open for the process lifetime; without this they leak on shutdown.
    """
    for agent_id, (instance, _graph) in list(_agent_registry.items()):
        runtime = getattr(instance, "context7_runtime", None)
        client = getattr(runtime, "client", None)
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
            logger.info("Closed Context7 MCP client for agent %s", agent_id)
        except Exception:
            logger.warning("Failed to close resources for agent %s", agent_id, exc_info=True)

    _agent_registry.clear()


def list_registered_agents() -> list[str]:
    """
    List all registered agent IDs.
    
    Returns:
        List of agent IDs that have factories registered
    """
    return list(_agent_factories.keys())
