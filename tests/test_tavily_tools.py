"""Tavily resolves credentials for each call without process environment changes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from juena.tools.tavily import load_optional_tavily_tool, tavily_search
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
    assert all(c["max_results"] == 2 and c["include_raw_content"] == "markdown" for c in calls)


@pytest.mark.asyncio
async def test_missing_key_prevents_tavily_client_construction(monkeypatch):
    import langchain_tavily
    monkeypatch.setattr(credentials, "personal_key", AsyncMock(side_effect=credentials.CredentialError("Add a key")))
    monkeypatch.setattr(langchain_tavily, "TavilySearch", lambda **kw: pytest.fail("No request allowed"))
    result = await tavily_search.coroutine("query", SimpleNamespace(context=SimpleNamespace(user_id="alice")))
    assert result == {"error": "Add a key"}
