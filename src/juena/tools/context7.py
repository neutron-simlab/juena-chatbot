"""Optional Context7 MCP tool loading for external docs/examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from juena_core.log import get_logger
from juena_core.mcp import discover_optional_tools
from juena.core.paths import get_config

logger = get_logger(__name__)

CONTEXT7_LABEL = "Context7"


@dataclass
class Context7Runtime:
    """The tools this integration contributed to the supervisor, if any.

    It no longer holds a client. `juena_core.mcp` returns tools that each
    reopen their own connection per call and survive an MCP server restart, so
    there is nothing whose lifetime has to match the agent's -- and nothing for
    `shutdown_agents` to close. The retired `MultiServerMCPClient` exposed
    neither `close` nor `aclose` either, so that shutdown was already a no-op.
    """

    tools: list[Any]


async def load_optional_context7_tools() -> Context7Runtime | None:
    """Return loaded Context7 tools or ``None`` when disabled/unavailable.

    Context7 is nobody's product: if it is unreachable its tools should not
    exist, rather than exist and fail the first time a model reaches for one.
    """

    config = get_config()
    api_key = (config.CONTEXT7_API_KEY or "").strip()
    if not api_key:
        return None

    # FastMCP's `MCPConfig` shape. The transport is inferred from the entry --
    # there is no `transport` key any more, and the timeout is seconds.
    tools = await discover_optional_tools(
        {
            "mcpServers": {
                "context7": {
                    "url": config.CONTEXT7_MCP_URL,
                    "headers": {"CONTEXT7_API_KEY": api_key},
                    "timeout": config.CONTEXT7_TIMEOUT_SECONDS,
                }
            }
        },
        label=CONTEXT7_LABEL,
    )
    if tools is None:
        return None

    logger.info("Loaded %d Context7 MCP tool(s)", len(tools))
    return Context7Runtime(tools=tools)
