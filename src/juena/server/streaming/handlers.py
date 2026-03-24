"""
Stream handlers for processing LangGraph streaming events.

This module provides handlers for different stream modes (updates, messages, custom)
to process and convert LangGraph streaming events into chat messages.
"""

from typing import Any, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.types import Interrupt, Overwrite
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from juena.server.utils import (
    convert_message_content_to_string,
    remove_tool_calls,
)
import json


class UpdatesStreamHandler:
    """Handler for stream_mode='updates' events."""
    
    def __init__(
        self,
        agent: CompiledStateGraph,
        config: RunnableConfig,
        run_id: str,
        user_input_message: str
    ):
        self.agent = agent
        self.config = config
        self.run_id = run_id
        self.user_input_message = user_input_message

    @staticmethod
    def _normalize_update_messages(update_messages: Any) -> list[BaseMessage | tuple]:
        """Normalize LangGraph update payloads into a message list."""

        if isinstance(update_messages, Overwrite):
            update_messages = update_messages.value

        if update_messages is None:
            return []
        if isinstance(update_messages, list):
            return update_messages
        if isinstance(update_messages, (BaseMessage, tuple)):
            return [update_messages]
        return list(update_messages) if hasattr(update_messages, "__iter__") else []
    
    def process_updates(
        self,
        event: dict[str, Any]
    ) -> list[BaseMessage]:
        """
        Process updates stream events and extract messages.
        
        Args:
            event: The updates event dictionary
            
        Returns:
            List of messages extracted from updates
        """
        new_messages = []
        
        for node, updates in event.items():
            # Handle interrupts
            if node == "__interrupt__":
                interrupt: Interrupt
                for interrupt in updates:
                    interrupt_msg = AIMessage(content=interrupt.value)
                    new_messages.append(interrupt_msg)
                continue
            
            # Extract messages from updates
            updates = updates or {}
            update_messages = updates.get("messages", [])
            new_messages.extend(self._normalize_update_messages(update_messages))
        
        return new_messages


class MessagesStreamHandler:
    """Handler for stream_mode='messages' token streaming."""
    
    def __init__(
        self,
        run_id: str
    ):
        self.run_id = run_id
    
    def process_messages(
        self,
        event: tuple[BaseMessage, dict[str, Any]],
        user_input_message: str
    ) -> Optional[str]:
        """
        Process messages stream events and yield token chunks.
        
        Args:
            event: Tuple of (message, metadata)
            user_input_message: Original user input message to filter duplicates
            
        Returns:
            SSE data string with token chunk, or None if should be skipped
        """
        msg, metadata = event
        
        # Skip messages with skip_stream tag
        if "skip_stream" in metadata.get("tags", []):
            return None
        
        # Only process AIMessageChunk for token streaming
        if not isinstance(msg, AIMessageChunk):
            return None
        
        content = remove_tool_calls(msg.content)
        if not content:
            return None
        
        token_content = convert_message_content_to_string(content)
        return f"data: {json.dumps({'type': 'token', 'content': token_content})}\n\n"


class CustomStreamHandler:
    """Handler for stream_mode='custom' events."""
    
    @staticmethod
    def process_custom(event: Any) -> list[BaseMessage]:
        """
        Process custom stream events.
        
        Args:
            event: The custom event data
            
        Returns:
            List containing the custom event as a message
        """
        return [event]
