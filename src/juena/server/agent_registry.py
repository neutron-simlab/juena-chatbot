"""
Agent registry for managing agent instances.

This module provides functions for creating, retrieving, and restarting agents.
Supports multiple agent types through a factory pattern. Users must register
their own agent factories before using agents.
"""
from typing import Any, Callable, Optional, Awaitable
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.core.config import global_config
from juena.schema.llm_models import Provider, get_default_model_for_provider
from juena.server.errors import AgentNotFoundError

logger = get_logger(__name__)

# Simple in-memory agent registry
# Key format: agent_id -> tuple(AgentInstance, CompiledStateGraph)
# Provider/model are now handled per-invocation via config.configurable
# Storing both agent instance and app allows us to restart the graph with new config
DEFAULT_AGENT = "react_agent"

# Type alias for agent instance (can be any type)
AgentInstance = Any

# Registry storage: agent_id -> (AgentInstance, CompiledStateGraph)
_agent_registry: dict[str, tuple[AgentInstance, CompiledStateGraph]] = {}

# Agent factory registry: agent_id -> async factory function
# Factory function signature: async (provider: str, model: str) -> tuple[AgentInstance, CompiledStateGraph]
_agent_factories: dict[str, Callable[[str, str], Awaitable[tuple[AgentInstance, CompiledStateGraph]]]] = {}

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
    After creation, model selection can be handled per invocation via config.configurable.
    
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


async def restart_agent(
    agent_id: str,
    provider: str | None = None,
    model: str | None = None
) -> CompiledStateGraph:
    """
    Restart an agent by clearing its state/memory.
    
    This function clears conversation state/memory for a fresh start.
    The graph itself is reused; model selection happens per-invocation via config.
    
    Args:
        agent_id: Agent identifier (e.g., "my_agent")
        provider: Ignored (kept for backward compatibility). Model selection is per-invocation.
        model: Ignored (kept for backward compatibility). Model selection is per-invocation.
    
    Returns:
        CompiledStateGraph for the agent (same graph, cleared state)
        
    Raises:
        AgentNotFoundError: If agent_id is not registered
    """
    logger.info(f"Restarting agent: {agent_id} (clearing state/memory)")
    
    # Check if factory is registered
    if agent_id not in _agent_factories:
        logger.error(f"Unknown agent requested: {agent_id}")
        available_agents = list(_agent_factories.keys())
        raise AgentNotFoundError(
            agent_id,
            details={"available_agents": available_agents}
        )
    
    # If agent exists, clear its state
    if agent_id in _agent_registry:
        agent_instance, _ = _agent_registry[agent_id]
        
        # Check if agent has restart_with_new_config method
        if hasattr(agent_instance, 'restart_with_new_config'):
            # Normalize provider/model for restart_with_new_config
            provider, model = _normalize_provider_model(provider, model)
            logger.info(f"Clearing state for {agent_id}")
            await agent_instance.restart_with_new_config(provider=provider, model=model, clear_state=True)
            # Update the registry with the new app (same graph, cleared state)
            _agent_registry[agent_id] = (agent_instance, agent_instance.app)
            logger.info(f"{agent_id} state cleared successfully")
        else:
            # Agent doesn't support restart, recreate it
            logger.info(f"{agent_id} doesn't support restart, recreating")
            # Remove old entry
            del _agent_registry[agent_id]
            # Create new one (provider/model used for initial default LLM)
            provider, model = _normalize_provider_model(provider, model)
            return await get_agent(agent_id, provider=provider, model=model)
    else:
        # Agent doesn't exist, create it
        logger.info(f"Agent not found, creating new {agent_id}")
        provider, model = _normalize_provider_model(provider, model)
        return await get_agent(agent_id, provider=provider, model=model)
    
    return _agent_registry[agent_id][1]


def get_agent_instance(agent_id: str = DEFAULT_AGENT) -> Optional[AgentInstance]:
    """
    Get the agent instance from the registry.
    
    This is useful for accessing agent-specific methods without needing to create a new agent.
    
    Args:
        agent_id: Agent identifier (defaults to DEFAULT_AGENT)
    
    Returns:
        Agent instance if found, None otherwise
    """
    if agent_id in _agent_registry:
        agent_instance, _ = _agent_registry[agent_id]
        return agent_instance
    return None


def list_registered_agents() -> list[str]:
    """
    List all registered agent IDs.
    
    Returns:
        List of agent IDs that have factories registered
    """
    return list(_agent_factories.keys())
