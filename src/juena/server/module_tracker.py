"""
Module tracking utilities for detecting the current module from state and node paths.

This module provides utilities for extracting module information from LangGraph
state and node paths, enabling proper module labeling for color coding in the UI.
"""

from typing import Any, Optional
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from juena.core.log import get_logger

logger = get_logger(__name__)


class ModuleTracker:
    """
    Utility class for tracking the current module from state and node paths.
    
    Extracts module information from:
    1. State's current_module field
    2. Node paths (e.g., "agent_node" -> "agent")
    3. Default fallback
    """
    
    @staticmethod
    async def get_current_module(
        agent: CompiledStateGraph,
        config: RunnableConfig,
        node_path: str | None = None
    ) -> Optional[str]:
        """
        Get the current module from state or node path.
        
        Args:
            agent: The compiled state graph
            config: The runnable config for state access
            node_path: Optional node path to extract module from
            
        Returns:
            Module name or None if not detected
        """
        # Try to get module from state first
        try:
            state: Any = await agent.aget_state(config=config)
            if state and hasattr(state, 'values') and state.values:
                current_module = state.values.get('current_module')
                if current_module:
                    return current_module
        except Exception as e:
            logger.debug(f"Could not get module from state: {e}")
        
        # Fallback to node path extraction
        if node_path:
            return ModuleTracker._extract_from_node_path(node_path)
        
        return None
    
    @staticmethod
    def _extract_from_node_path(node_path: str) -> Optional[str]:
        """
        Extract module name from node path dynamically.
        
        This method extracts the module name by splitting on underscore and taking
        the first part. This allows any module name to be detected dynamically.
        
        Examples:
            "agent_node" -> "agent"
            "my_agent" -> "my"
            "chatbot" -> "chatbot"
            
        Args:
            node_path: The node path string
            
        Returns:
            Module name or None if not detected
        """
        if not isinstance(node_path, str):
            return None
        
        # Extract module from path parts - take first part as module name
        parts = node_path.split('_')
        if parts and parts[0]:
            return parts[0]
        
        return None
    
    @staticmethod
    def get_module_for_message(
        current_module: Optional[str],
        node_path: Optional[str],
        default: str = 'default'
    ) -> str:
        """
        Get module name for a message, with fallback logic.
        
        Args:
            current_module: Current module from state
            node_path: Node path for extraction
            default: Default module name if none detected
            
        Returns:
            Module name to use for the message
        """
        if current_module:
            return current_module
        
        if node_path:
            extracted = ModuleTracker._extract_from_node_path(node_path)
            if extracted:
                return extracted
        
        return default
    
    @staticmethod
    def is_internal_node(node_path: str) -> bool:
        """Check if a node path represents an internal (non-user-facing) node."""
        if not isinstance(node_path, str):
            return False
        
        # Internal nodes that don't emit user-facing messages
        # Users can customize this list for their own internal nodes
        internal_keywords = []
        return any(keyword in node_path for keyword in internal_keywords)
