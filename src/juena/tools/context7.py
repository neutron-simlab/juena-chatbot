"""Optional Context7 MCP tool loading for external docs/examples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from juena.core.log import get_logger

logger = get_logger(__name__)


def _get_config() -> Any:
    """Lazy-load config to keep imports testable."""
    from juena.core.config import Config

    return Config


def _get_mcp_client_class() -> type[Any]:
    """Import the MCP client lazily so local repo chat still works without it."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient


@dataclass
class Context7Runtime:
    """Keep the MCP client alive for as long as the agent is registered."""

    client: Any
    tools: list[Any]


async def load_optional_context7_tools() -> Context7Runtime | None:
    """Return loaded Context7 tools or ``None`` when disabled/unavailable."""

    config = _get_config()
    api_key = (config.CONTEXT7_API_KEY or "").strip()
    if not api_key:
        return None

    try:
        client_cls = _get_mcp_client_class()
    except ImportError as exc:
        logger.warning(
            "Context7 requested but langchain-mcp-adapters is unavailable: %s",
            exc,
        )
        return None

    try:
        client = client_cls(
            {
                "context7": {
                    "transport": "streamable_http",
                    "url": config.CONTEXT7_MCP_URL,
                    "headers": {"CONTEXT7_API_KEY": api_key},
                    "timeout": timedelta(seconds=config.CONTEXT7_TIMEOUT_SECONDS),
                }
            }
        )
        tools = list(await client.get_tools())
    except Exception as exc:
        logger.warning("Failed to load Context7 MCP tools: %s", exc)
        return None

    logger.info("Loaded %d Context7 MCP tool(s)", len(tools))
    return Context7Runtime(client=client, tools=tools)
