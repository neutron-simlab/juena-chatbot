"""
Agent input preparation utilities for LangGraph agents.

This module provides utilities for preparing input for LangGraph agent execution.
LangGraph automatically resumes from checkpoint when invoked with the same thread_id.
"""

from typing import Any
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.agent_registry import _normalize_provider_model

logger = get_logger(__name__)


class AgentInputHandler:
    """Handles input preparation for LangGraph agents."""
    
    @staticmethod
    async def prepare_input(
        user_input: str,
        agent: CompiledStateGraph,
        thread_id: str | None = None,
        user_id: str | None = None,
        run_id: UUID | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[dict[str, Any], UUID]:
        """
        Prepare input for agent invocation.
        
        LangGraph automatically resumes from checkpoint when invoked with same thread_id.
        Provider and model are placed in config.configurable so dynamic model
        middleware can use them without restarting the graph.
        
        Args:
            user_input: User input message
            agent: The compiled state graph
            thread_id: Optional thread ID for conversation continuity
            user_id: Optional user ID for cross-thread conversations
            run_id: Optional run ID, will generate if not provided
            provider: Optional LLM provider (openai/blablador); normalized to default if None
            model: Optional LLM model name; normalized to provider default if None
            
        Returns:
            Tuple of (kwargs for agent invocation, run_id)
            
        Raises:
            StateError: If there's an error accessing agent state
        """
        run_id = run_id or uuid4()
        thread_id = thread_id or str(uuid4())
        user_id = user_id or str(uuid4())
        provider, model = _normalize_provider_model(provider, model)
        
        configurable: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "provider": provider,
            "model": model,
        }
        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
        )
        
        # Prepare input - always add as human message
        # LangGraph will automatically resume from checkpoint
        input_data = {
            "messages": [HumanMessage(content=user_input)],
            "thread_id": thread_id,
            "user_id": user_id,
        }
        
        kwargs = {
            "input": input_data,
            "config": config,
        }
        
        return kwargs, run_id
