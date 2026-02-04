"""
Centralized logging configuration for Chatbot Template.

This module provides reusable logging setup functions to eliminate code duplication
across the codebase. All loggers use consistent formatting and respect the LOG_LEVEL
configuration from the Config class.
"""
import logging
from typing import Union

from juena.core.config import Config

# Log format constants
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def _get_log_level(level: Union[int, str, None] = None) -> int:
    """
    Convert log level to logging constant.
    
    Args:
        level: Log level as int, string (e.g., "INFO", "DEBUG"), or None to use Config.LOG_LEVEL
        
    Returns:
        Logging level constant (e.g., logging.INFO, logging.DEBUG)
    """
    if level is None:
        level = Config.LOG_LEVEL
    
    if isinstance(level, int):
        return level
    
    if isinstance(level, str):
        level_upper = level.upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        return level_map.get(level_upper, logging.INFO)
    
    # Fallback (should never be reached due to type hints, but kept for safety)
    return logging.INFO


def setup_logger(logger: logging.Logger, level: Union[int, str, None] = None) -> None:
    """
    Setup a logger with handler, formatter, and level configuration.
    
    This function configures the logger only if it doesn't already have handlers,
    preventing duplicate log messages.
    
    Args:
        logger: The logger instance to configure
        level: Log level as int, string, or None to use Config.LOG_LEVEL
    """
    # Only add handler if logger doesn't have one (avoid duplicates)
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(_get_log_level(level))
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(_get_log_level(level))
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False


def get_logger(name: str, level: Union[int, str, None] = None) -> logging.Logger:
    """
    Get or create a logger with the given name and configure it.
    
    This is the main function to use for getting configured loggers throughout
    the codebase. It ensures consistent logging setup across all modules.
    
    Args:
        name: Logger name (typically __name__ or a module path)
        level: Log level as int, string, or None to use Config.LOG_LEVEL
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from juena.core.log import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    logger = logging.getLogger(name)
    setup_logger(logger, level)
    return logger
