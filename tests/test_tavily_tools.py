"""Tests for optional Tavily tool loading."""

from __future__ import annotations

from typing import Any

import pytest

from juena.tools import tavily as tavily_module


def test_load_optional_tavily_tool_returns_none_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubConfig:
        TAVILY_API_KEY = None

    monkeypatch.setattr(tavily_module, "_get_config", lambda: StubConfig)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    tool = tavily_module.load_optional_tavily_tool()

    assert tool is None


def test_load_optional_tavily_tool_logs_warning_and_returns_none_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    class StubConfig:
        TAVILY_API_KEY = "tvly-key"

    def fake_warning(message: str, *args: Any) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(tavily_module, "_get_config", lambda: StubConfig)
    monkeypatch.setattr(
        tavily_module,
        "_get_tavily_search_class",
        lambda: (_ for _ in ()).throw(ImportError("missing package")),
    )
    monkeypatch.setattr(tavily_module.logger, "warning", fake_warning)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    tool = tavily_module.load_optional_tavily_tool()

    assert tool is None
    assert warnings == ["Tavily requested but langchain-tavily is unavailable: missing package"]


def test_load_optional_tavily_tool_logs_warning_and_returns_none_on_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    class StubConfig:
        TAVILY_API_KEY = "tvly-key"

    class FakeTavilySearch:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    def fake_warning(message: str, *args: Any) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(tavily_module, "_get_config", lambda: StubConfig)
    monkeypatch.setattr(tavily_module, "_get_tavily_search_class", lambda: FakeTavilySearch)
    monkeypatch.setattr(tavily_module.logger, "warning", fake_warning)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    tool = tavily_module.load_optional_tavily_tool()

    assert tool is None
    assert warnings == ["Failed to initialize Tavily search tool: boom"]


def test_load_optional_tavily_tool_returns_compact_search_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class StubConfig:
        TAVILY_API_KEY = "tvly-key"

    class FakeTavilySearch:
        def __init__(self, **kwargs: Any) -> None:
            created["kwargs"] = kwargs

    monkeypatch.setattr(tavily_module, "_get_config", lambda: StubConfig)
    monkeypatch.setattr(tavily_module, "_get_tavily_search_class", lambda: FakeTavilySearch)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    tool = tavily_module.load_optional_tavily_tool()

    assert isinstance(tool, FakeTavilySearch)
    assert created["kwargs"] == {
        "max_results": tavily_module.TAVILY_MAX_RESULTS,
        "topic": tavily_module.TAVILY_TOPIC,
        "search_depth": tavily_module.TAVILY_SEARCH_DEPTH,
        "include_answer": tavily_module.TAVILY_INCLUDE_ANSWER,
        "include_raw_content": tavily_module.TAVILY_INCLUDE_RAW_CONTENT,
        "include_images": tavily_module.TAVILY_INCLUDE_IMAGES,
        "include_image_descriptions": tavily_module.TAVILY_INCLUDE_IMAGE_DESCRIPTIONS,
    }
