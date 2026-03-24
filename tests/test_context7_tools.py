"""Tests for optional Context7 MCP tool loading."""

from __future__ import annotations

from datetime import timedelta
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

    monkeypatch.setattr(context7_module, "_get_config", lambda: StubConfig)

    runtime = await context7_module.load_optional_context7_tools()

    assert runtime is None


@pytest.mark.asyncio
async def test_load_optional_context7_tools_loads_remote_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_a = object()
    tool_b = object()
    created: dict[str, Any] = {}

    class StubConfig:
        CONTEXT7_API_KEY = "ctx-key"
        CONTEXT7_MCP_URL = "https://example.com/mcp"
        CONTEXT7_TIMEOUT_SECONDS = 45

    class FakeMultiServerMCPClient:
        def __init__(self, config: dict[str, Any]) -> None:
            created["config"] = config

        async def get_tools(self) -> list[object]:
            return [tool_a, tool_b]

    monkeypatch.setattr(context7_module, "_get_config", lambda: StubConfig)
    monkeypatch.setattr(context7_module, "_get_mcp_client_class", lambda: FakeMultiServerMCPClient)

    runtime = await context7_module.load_optional_context7_tools()

    assert runtime is not None
    assert runtime.tools == [tool_a, tool_b]
    assert isinstance(runtime.client, FakeMultiServerMCPClient)
    assert created["config"] == {
        "context7": {
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "headers": {"CONTEXT7_API_KEY": "ctx-key"},
            "timeout": timedelta(seconds=45),
        }
    }


@pytest.mark.asyncio
async def test_load_optional_context7_tools_logs_warning_and_returns_none_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    class StubConfig:
        CONTEXT7_API_KEY = "ctx-key"
        CONTEXT7_MCP_URL = "https://example.com/mcp"
        CONTEXT7_TIMEOUT_SECONDS = 30

    class FakeMultiServerMCPClient:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        async def get_tools(self) -> list[object]:
            raise RuntimeError("boom")

    def fake_warning(message: str, *args: Any) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(context7_module, "_get_config", lambda: StubConfig)
    monkeypatch.setattr(context7_module, "_get_mcp_client_class", lambda: FakeMultiServerMCPClient)
    monkeypatch.setattr(context7_module.logger, "warning", fake_warning)

    runtime = await context7_module.load_optional_context7_tools()

    assert runtime is None
    assert warnings == ["Failed to load Context7 MCP tools: boom"]
