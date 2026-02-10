"""
Message processor for converting LangChain messages to chat format for streaming.

This module processes messages and converts them to chat format, handling
deduplication and filtering.
"""

import inspect
import json
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from juena.core.log import get_logger
from juena.server.utils import langchain_to_chat_message
from juena.server.errors import MessageProcessingError
from juena.server.streaming.deduplication import get_message_identifier

logger = get_logger(__name__)


class MessageProcessor:
    """Processes and converts messages to chat format for streaming."""
    
    def __init__(
        self,
        agent: CompiledStateGraph,
        config: RunnableConfig,
        run_id: str,
        user_input_message: str,
        streamed_message_ids: set[str]
    ):
        self.agent = agent
        self.config = config
        self.run_id = run_id
        self.user_input_message = user_input_message
        self.streamed_message_ids = streamed_message_ids
    
    def _create_ai_message(self, parts: dict) -> AIMessage:
        """Create an AIMessage from parts dictionary."""
        sig = inspect.signature(AIMessage)
        valid_keys = set(sig.parameters)
        filtered = {k: v for k, v in parts.items() if k in valid_keys}
        return AIMessage(**filtered)
    
    def _process_message_parts(self, messages: list[BaseMessage | tuple]) -> list[BaseMessage]:
        """
        Process messages that may contain tuples (field_name, field_value).
        
        LangGraph streaming may emit tuples: (field_name, field_value)
        e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), etc.
        We accumulate these into complete messages.
        
        Args:
            messages: List of messages that may contain tuples
            
        Returns:
            List of processed BaseMessage objects
        """
        processed_messages = []
        current_message: dict[str, Any] = {}
        
        for message in messages:
            if isinstance(message, tuple):
                key, value = message
                current_message[key] = value
            else:
                # Complete message - add any accumulated parts first
                if current_message:
                    processed_messages.append(self._create_ai_message(current_message))
                    current_message = {}
                processed_messages.append(message)
        
        # Add any remaining message parts
        if current_message:
            processed_messages.append(self._create_ai_message(current_message))
        
        return processed_messages
    
    async def process_and_yield_messages(
        self,
        messages: list[BaseMessage]
    ) -> AsyncGenerator[str, None]:
        """
        Process messages and yield SSE-formatted strings.
        
        Deduplicates messages at the backend level before streaming to prevent
        duplicate messages from multiple stream modes (updates, messages, custom).
        
        Args:
            messages: List of messages to process
            
        Yields:
            SSE-formatted data strings
        """
        # Process message parts (handle tuple-based streaming)
        processed_messages = self._process_message_parts(messages)
        
        for message in processed_messages:
            try:
                # Skip SystemMessage - they should remain in state but not be streamed
                if isinstance(message, SystemMessage):
                    continue
                
                # Convert to ChatMessage
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(self.run_id)
                
                # Filter out duplicate user input messages
                if chat_message.type == "human" and chat_message.content == self.user_input_message:
                    continue
                
                # Skip system messages
                if chat_message.type == "system":
                    continue
                
                # Skip AIMessages with empty content that only have tool_calls
                # These are tool invocation messages - the tool result will be shown separately
                if (chat_message.type == "ai" and 
                    chat_message.tool_calls and
                    (not chat_message.content or (isinstance(chat_message.content, str) and chat_message.content.strip() == ""))):
                    tool_names = [tc.get('name', 'unknown') for tc in chat_message.tool_calls]
                    logger.debug(f"Skipping empty AIMessage with tool_calls: {tool_names}")
                    continue
                
                # Deduplication: Check if we've already streamed this message
                message_id = get_message_identifier(
                    message, 
                    chat_message,
                    run_id=self.run_id
                )
                if message_id in self.streamed_message_ids:
                    # Skip duplicate message
                    continue
                
                # Mark message as streamed
                self.streamed_message_ids.add(message_id)
                
                # Yield SSE-formatted message
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                error_msg = MessageProcessingError(
                    "Failed to process message",
                    message_type=type(message).__name__,
                    details={"error": str(e)}
                )
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg.message})}\n\n"
