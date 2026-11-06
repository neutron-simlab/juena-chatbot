"""Tavily resolves credentials for each call without process environment changes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from juena.tools.tavily import load_optional_tavily_tool, tavily_extract, tavily_search
from juena.server import credentials


@pytest.mark.asyncio
async def test_personal_tavily_keys_are_explicit_and_refresh_each_call(monkeypatch):
    import langchain_tavily
    calls = []
    key = AsyncMock(side_effect=["fake-alice", "fake-bob", "fake-new-alice"])
    monkeypatch.setattr(credentials, "personal_key", key)
    monkeypatch.setenv("TAVILY_API_KEY", "must-not-use-server-key")
    class Search:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        async def ainvoke(self, query, **kwargs):
            return {"results": [{"content": "Evidence"}]}
    monkeypatch.setattr(langchain_tavily, "TavilySearch", Search)
    assert load_optional_tavily_tool() is tavily_search
    for owner in ("alice", "bob", "alice"):
        result = await tavily_search.coroutine("query", SimpleNamespace(context=SimpleNamespace(user_id=owner)))
        assert result["results"]
    assert [c["tavily_api_key"] for c in calls] == ["fake-alice", "fake-bob", "fake-new-alice"]
    assert all(c["max_results"] == 2 and c["include_raw_content"] is False for c in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,wrapper", [("TavilySearch", tavily_search), ("TavilyExtract", tavily_extract)])
async def test_missing_key_prevents_tavily_client_construction(monkeypatch, name, wrapper):
    import langchain_tavily
    monkeypatch.setattr(credentials, "personal_key", AsyncMock(side_effect=credentials.CredentialError("Add a key")))
    monkeypatch.setattr(langchain_tavily, name, lambda **kw: pytest.fail("No request allowed"))
    result = await wrapper.coroutine("query", SimpleNamespace(context=SimpleNamespace(user_id="alice")))
    assert result == ({"error": "Add a key"} if wrapper is tavily_search else "Add a key")


@pytest.mark.asyncio
async def test_extract_refreshes_personal_keys_and_preserves_table_headers(monkeypatch):
    from langchain_tavily._utilities import TavilyExtractAPIWrapper

    key = AsyncMock(side_effect=["fake-alice", "fake-bob", "fake-new-alice"])
    monkeypatch.setattr(credentials, "personal_key", key)
    monkeypatch.setenv("TAVILY_API_KEY", "must-not-use-server-key")
    url = "https://example.org/neutron-table"
    page = "| Element | b (fm) | sigma (barn) |\n|---|---|---|\n| C | 6.646 | 5.551 |"
    calls = []

    async def extract(client, **kwargs):
        calls.append((client.tavily_api_key.get_secret_value(), kwargs))
        return {"results": [{"url": url, "raw_content": page}]}

    monkeypatch.setattr(TavilyExtractAPIWrapper, "raw_results_async", extract)
    for owner, query in [("alice", "data policy"), ("bob", None), ("alice", "metadata")]:
        result = await tavily_extract.coroutine(
            url, SimpleNamespace(context=SimpleNamespace(user_id=owner)), query=query,
        )
        assert result == f"Source: {url}\n\n{page}"
        assert "fake-" not in result
    assert [key for key, _ in calls] == ["fake-alice", "fake-bob", "fake-new-alice"]
    assert [call["query"] for _, call in calls] == ["data policy", None, "metadata"]
    assert [call["chunks_per_source"] for _, call in calls] == [3, None, 3]
    assert all(call["urls"] == [url] and call["format"] == "markdown"
               and call["extract_depth"] == "advanced" for _, call in calls)


def test_extract_accepts_one_url_not_a_batch():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        tavily_extract.tool_call_schema.model_validate({"url": ["https://example.org/a", "https://example.org/b"]})


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"results": [], "failed_results": [{"url": "https://example.org", "error": "private upstream detail"}]},
    {"error": "private upstream detail"},
    "private upstream detail",
    RuntimeError("private upstream detail"),
])
async def test_extract_failures_do_not_expose_upstream_details(monkeypatch, response):
    import langchain_tavily

    monkeypatch.setattr(credentials, "personal_key", AsyncMock(return_value="fake-key"))
    call = AsyncMock(side_effect=response) if isinstance(response, Exception) else AsyncMock(return_value=response)
    monkeypatch.setattr(langchain_tavily, "TavilyExtract", lambda **kw: SimpleNamespace(ainvoke=call))
    result = await tavily_extract.coroutine(
        "https://example.org", SimpleNamespace(context=SimpleNamespace(user_id="alice")),
    )
    assert result.startswith("Tavily could not read this page.")
    assert "private upstream detail" not in result
