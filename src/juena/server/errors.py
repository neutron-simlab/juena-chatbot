"""
Custom exceptions for the Chatbot Template server.

This module defines custom exception classes for better error handling
and more specific error messages throughout the service layer.
"""

from typing import Any, Optional


class ChatbotServerError(Exception):
    """Base exception for all chatbot server errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentNotFoundError(ChatbotServerError):
    """Raised when a requested agent is not found or cannot be created."""
    
    def __init__(self, agent_id: str, details: dict[str, Any] | None = None):
        message = f"Agent '{agent_id}' not found or could not be created"
        super().__init__(message, details)
        self.agent_id = agent_id


class StreamingError(ChatbotServerError):
    """Raised when an error occurs during streaming operations."""
    
    def __init__(self, message: str, stream_mode: Optional[str] = None, details: dict[str, Any] | None = None):
        super().__init__(message, details)
        self.stream_mode = stream_mode


class StateError(ChatbotServerError):
    """Raised when an error occurs during state operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: dict[str, Any] | None = None):
        super().__init__(message, details)
        self.operation = operation


class MessageProcessingError(ChatbotServerError):
    """Raised when an error occurs during message processing."""
    
    def __init__(self, message: str, message_type: Optional[str] = None, details: dict[str, Any] | None = None):
        super().__init__(message, details)
        self.message_type = message_type
