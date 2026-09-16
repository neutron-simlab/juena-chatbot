"""Tests for optional Context7 MCP tool loading.

The transport moved to `juena_core.mcp` during the plan-02 cutover, so what is
this application's -- and what these tests cover -- is the *config it builds*
and the policy that a missing key or an unreachable server costs a capability
rather than raising. The `MCPAdapter` lifecycle itself is measured in core's
`tests/cp0b/` against a real FastMCP server.
"""

from __future__ import annotations

from typing import Any

import pytest

from juena.tools import context7 as context7_module


@pytest.mark.asyncio
async def test_load_optional_context7_tools_returns_none_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubConfig:
        CONTEXT7_API_KEY = None
        CONTEXT7_MCP_URL = "https://mcp.context7.com/mcp"
        CONTEXT7_TIMEOUT_SECONDS = 30

    monkeypatch.setattr(context7_module, "get_config", lambda: StubConfig)

    runtime = await context7_module.load_optional_context7_tools()

    assert runtime is None


@pytest.mark.asyncio
async def test_load_optional_context7_tools_loads_remote_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_a = object()
    tool_b = object()
    discovered: dict[str, Any] = {}

    class StubConfig:
        CONTEXT7_API_KEY = "ctx-key"
        CONTEXT7_MCP_URL = "https://example.com/mcp"
        CONTEXT7_TIMEOUT_SECONDS = 45

    async def fake_discover(target: Any, *, label: str) -> list[object]:
        discovered["target"] = target
        discovered["label"] = label
        return [tool_a, tool_b]

    monkeypatch.setattr(context7_module, "get_config", lambda: StubConfig)
    monkeypatch.setattr(context7_module, "discover_optional_tools", fake_discover)

    runtime = await context7_module.load_optional_context7_tools()

    assert runtime is not None
    assert runtime.tools == [tool_a, tool_b]
    assert discovered["label"] == "Context7"
    # FastMCP's `MCPConfig`: servers under `mcpServers`, the transport inferred
    # from the entry rather than named, and a plain number of seconds.
    assert discovered["target"] == {
        "mcpServers": {
            "context7": {
                "url": "https://example.com/mcp",
                "headers": {"CONTEXT7_API_KEY": "ctx-key"},
                "timeout": 45,
            }
        }
    }


@pytest.mark.asyncio
async def test_an_unreachable_context7_costs_a_capability_not_a_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional-integration policy, which is this application's choice.

    Core offers both: `discover_tools` raises for an integration that *is* the
    product, `discover_optional_tools` returns None for one that is not.
    Context7 is the second, so the supervisor is built with one tool fewer.
    """

    class StubConfig:
        CONTEXT7_API_KEY = "ctx-key"
        CONTEXT7_MCP_URL = "https://example.com/mcp"
        CONTEXT7_TIMEOUT_SECONDS = 30

    async def unreachable(target: Any, *, label: str) -> None:
        return None

    monkeypatch.setattr(context7_module, "get_config", lambda: StubConfig)
    monkeypatch.setattr(context7_module, "discover_optional_tools", unreachable)

    assert await context7_module.load_optional_context7_tools() is None
