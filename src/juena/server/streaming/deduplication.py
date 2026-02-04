"""
Message deduplication utilities for streaming.

This module provides utilities for generating unique message identifiers
to prevent duplicate messages from being streamed.
"""

import hashlib
from typing import Any, Optional

from langchain_core.messages import BaseMessage


def get_message_identifier(
    message: BaseMessage,
    chat_message: Any,
    module_name: Optional[str],
    run_id: str | None = None
) -> str:
    """
    Get a unique identifier for a message for deduplication.
    
    Uses message.id if available, otherwise creates a hash from
    content, type, run_id, and module_name.
    
    Args:
        message: LangChain BaseMessage
        chat_message: Converted ChatMessage
        module_name: Optional module name
        run_id: Optional run ID
        
    Returns:
        Unique identifier string
    """
    # Try to use LangChain message ID first (most reliable)
    if hasattr(message, 'id') and message.id:
        return str(message.id)
    
    # Fallback: create hash from message attributes
    content = chat_message.content or ""
    msg_type = chat_message.type or "unknown"
    run_id_str = str(run_id) if run_id else ""
    module = module_name or ""
    
    # Create hash from key attributes
    hash_input = f"{msg_type}:{content}:{run_id_str}:{module}"
    return hashlib.md5(hash_input.encode()).hexdigest()
