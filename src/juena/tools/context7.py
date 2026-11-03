"""Context7 tools with request-owned MCP credentials."""

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain.tools import ToolRuntime, tool
from juena_core.mcp import call_personal_tool
from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext
from juena.core.paths import get_config


@dataclass
class Context7Runtime:
    tools: list[Any]


async def call_context7(key: str, name: str, arguments: dict) -> Any:
    config = get_config()
    async with asyncio.timeout(config.CONTEXT7_TIMEOUT_SECONDS):
        return await call_personal_tool({"mcpServers": {"context7": {
            "url": config.CONTEXT7_MCP_URL,
            "headers": {"CONTEXT7_API_KEY": key},
            "timeout": config.CONTEXT7_TIMEOUT_SECONDS,
        }}}, name, arguments)


async def _call(user_id: str, name: str, arguments: dict) -> Any:
    from juena.server.credentials import CredentialError, personal_key
    try:
        key = await personal_key(user_id, "context7")
        return await call_context7(key, name, arguments)
    except CredentialError as error:
        return {"error": str(error)}
    except Exception:
        return {"error": "Context7 could not complete the lookup. Check your key in My profile or retry later."}


@tool("resolve-library-id")
async def resolve_library_id(libraryName: str, query: str, runtime: ToolRuntime[RuntimeModelContext]) -> Any:
    """Find a Context7 library ID before fetching its documentation."""
    return await _call(runtime.context.user_id, "resolve-library-id", {"libraryName": libraryName, "query": query})


@tool("query-docs")
async def query_docs(libraryId: str, query: str, runtime: ToolRuntime[RuntimeModelContext], language: str = "") -> Any:
    """Fetch documentation for a resolved Context7 library ID and question."""
    arguments = {"libraryId": libraryId, "query": query}
    if language:
        arguments["language"] = language
    return await _call(runtime.context.user_id, "query-docs", arguments)


async def load_optional_context7_tools() -> Context7Runtime:
    # No network or credentials during process-wide graph construction.
    return Context7Runtime(tools=[resolve_library_id, query_docs])
