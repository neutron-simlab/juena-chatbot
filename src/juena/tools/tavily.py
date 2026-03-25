"""Optional Tavily web-search tool loading."""

from __future__ import annotations

import os
from typing import Any

from juena.core.log import get_logger

logger = get_logger(__name__)

TAVILY_MAX_RESULTS = 3
TAVILY_TOPIC = "general"
TAVILY_SEARCH_DEPTH = "basic"
TAVILY_INCLUDE_ANSWER = False
TAVILY_INCLUDE_RAW_CONTENT = False
TAVILY_INCLUDE_IMAGES = False
TAVILY_INCLUDE_IMAGE_DESCRIPTIONS = False


def _get_config() -> Any:
    """Lazy-load config to keep imports testable."""
    from juena.core.config import Config

    return Config


def _get_tavily_search_class() -> type[Any]:
    """Import Tavily lazily so the app still works when the package is absent."""
    from langchain_tavily import TavilySearch

    return TavilySearch


def load_optional_tavily_tool() -> Any | None:
    """Return a configured Tavily search tool or ``None`` when disabled."""

    config = _get_config()
    api_key = (config.TAVILY_API_KEY or "").strip()
    if not api_key:
        return None

    if not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = api_key

    try:
        tavily_cls = _get_tavily_search_class()
    except ImportError as exc:
        logger.warning(
            "Tavily requested but langchain-tavily is unavailable: %s",
            exc,
        )
        return None

    try:
        return tavily_cls(
            max_results=TAVILY_MAX_RESULTS,
            topic=TAVILY_TOPIC,
            search_depth=TAVILY_SEARCH_DEPTH,
            include_answer=TAVILY_INCLUDE_ANSWER,
            include_raw_content=TAVILY_INCLUDE_RAW_CONTENT,
            include_images=TAVILY_INCLUDE_IMAGES,
            include_image_descriptions=TAVILY_INCLUDE_IMAGE_DESCRIPTIONS,
        )
    except Exception as exc:
        logger.warning("Failed to initialize Tavily search tool: %s", exc)
        return None
