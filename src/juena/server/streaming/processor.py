"""
Main stream event processor for LangGraph streaming events.

This module provides the main StreamEventProcessor class that orchestrates
the processing of different stream modes.
"""

import json
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from juena.core.log import get_logger
from juena.server.module_tracker import ModuleTracker
from juena.server.errors import StreamingError
from juena.server.utils import langchain_to_chat_message
from juena.server.streaming.handlers import (
    UpdatesStreamHandler,
    MessagesStreamHandler,
    CustomStreamHandler,
)
from juena.server.streaming.message_processor import MessageProcessor
from juena.server.streaming.deduplication import get_message_identifier

logger = get_logger(__name__)


class StreamEventProcessor:
    """Main processor for LangGraph stream events."""
    
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
        
        # Track already-streamed messages to prevent duplicates
        # Uses message IDs (from LangChain BaseMessage.id) or hash fallback
        self._streamed_message_ids: set[str] = set()
        
        # Initialize handlers
        self.updates_handler = UpdatesStreamHandler(agent, config, run_id, user_input_message)
        self.message_processor = MessageProcessor(
            agent, config, run_id, user_input_message, self._streamed_message_ids
        )
        
        # Track current module
        self.current_module: Optional[str] = None
    
    def _parse_stream_event(self, stream_event: Any) -> tuple[str, Any, Optional[str]]:
        """
        Parse a stream event into (stream_mode, event, node_path).
        
        Args:
            stream_event: The raw stream event from LangGraph
            
        Returns:
            Tuple of (stream_mode, event, node_path)
        """
        if not isinstance(stream_event, tuple):
            raise StreamingError(f"Unexpected stream event type: {type(stream_event)}")
        
        if len(stream_event) == 3:
            # With subgraphs=True: (node_path, stream_mode, event)
            node_path, stream_mode, event = stream_event
            return stream_mode, event, node_path
        else:
            # Without subgraphs: (stream_mode, event)
            stream_mode, event = stream_event
            return stream_mode, event, None
    
    async def _update_current_module(self, node_path: Optional[str]) -> None:
        """Update the current module from state or node path."""
        self.current_module = await ModuleTracker.get_current_module(
            self.agent,
            self.config,
            node_path
        )
    
    async def _initialize_streamed_message_ids(self) -> None:
        """
        Initialize streamed_message_ids by reading existing messages from state.
        
        This prevents duplicate messages from being streamed when the graph resumes
        from a checkpoint. All existing messages in the state are marked as already
        streamed, so only new messages will be sent to the client.
        
        Handles edge cases:
        - State doesn't exist yet (first invocation) - no messages to initialize
        - State exists but has no messages - empty set
        - Errors accessing state - logs warning and continues with empty set
        """
        try:
            # Get current state from the agent
            state: Any = await self.agent.aget_state(config=self.config)
            
            # Check if state exists and has values
            if not state or not hasattr(state, 'values') or not state.values:
                logger.debug("No existing state found, starting with empty streamed_message_ids")
                return
            
            # Extract messages from state
            existing_messages = state.values.get('messages', [])
            
            if not existing_messages:
                logger.debug("State exists but has no messages, starting with empty streamed_message_ids")
                return
            
            # Get current module from state for consistent module detection
            current_module_from_state = state.values.get('current_module')
            
            # Generate IDs for all existing messages
            initialized_count = 0
            for message in existing_messages:
                try:
                    # Skip SystemMessage - they shouldn't be streamed anyway
                    if isinstance(message, SystemMessage):
                        continue
                    
                    # Determine module for this message
                    # Use state's current_module if available, otherwise try to extract from message metadata
                    module_for_message = current_module_from_state
                    if not module_for_message and hasattr(message, 'additional_kwargs'):
                        module_for_message = message.additional_kwargs.get('module_name')
                    
                    # Convert to ChatMessage to get consistent format
                    chat_message = langchain_to_chat_message(
                        message, 
                        module_name=module_for_message
                    )
                    
                    # Generate message ID using same logic as MessageProcessor
                    message_id = get_message_identifier(
                        message,
                        chat_message,
                        module_for_message,
                        run_id=self.run_id
                    )
                    
                    # Add to streamed_message_ids
                    self._streamed_message_ids.add(message_id)
                    initialized_count += 1
                    
                except Exception as e:
                    # Log but continue processing other messages
                    logger.warning(f"Error initializing message ID for message: {e}", exc_info=True)
            
            logger.debug(f"Initialized streamed_message_ids with {initialized_count} existing messages")
            
        except Exception as e:
            # Log error but don't fail - we can still process new messages
            # This ensures the system is resilient to state access issues
            logger.warning(f"Failed to initialize streamed_message_ids from state: {e}", exc_info=True)
            # Continue with empty set - new messages will still be processed
    
    async def process_event(
        self,
        stream_event: Any
    ) -> AsyncGenerator[str, None]:
        """
        Process a single stream event and yield SSE-formatted strings.
        
        Args:
            stream_event: The raw stream event from LangGraph
            
        Yields:
            SSE-formatted data strings
        """
        try:
            stream_mode, event, node_path = self._parse_stream_event(stream_event)
            
            # Update current module
            await self._update_current_module(node_path)
            
            # Process based on stream mode
            if stream_mode == "updates":
                messages = self.updates_handler.process_updates(
                    event,
                    node_path,
                    self.current_module
                )
                async for sse_string in self.message_processor.process_and_yield_messages(
                    messages,
                    node_path,
                    self.current_module
                ):
                    yield sse_string
            
            elif stream_mode == "messages":
                handler = MessagesStreamHandler(self.run_id, self.current_module)
                token_data = handler.process_messages(event, self.user_input_message)
                if token_data:
                    yield token_data
            
            elif stream_mode == "custom":
                messages = CustomStreamHandler.process_custom(event)
                async for sse_string in self.message_processor.process_and_yield_messages(
                    messages,
                    node_path,
                    self.current_module
                ):
                    yield sse_string
            
        except Exception as e:
            logger.error(f"Error processing stream event: {e}", exc_info=True)
            error = StreamingError(
                "Failed to process stream event",
                stream_mode=getattr(e, 'stream_mode', None),
                details={"error": str(e)}
            )
            yield f"data: {json.dumps({'type': 'error', 'content': error.message})}\n\n"
