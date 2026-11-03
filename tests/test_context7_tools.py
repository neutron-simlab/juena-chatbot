"""Context7 credentials are scoped to a call, never the cached graph."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from juena.tools import context7
from juena.server import credentials


@pytest.mark.asyncio
async def test_context7_build_needs_no_key_or_network(monkeypatch):
    call = AsyncMock()
    monkeypatch.setattr(context7, "call_personal_tool", call)
    runtime = await context7.load_optional_context7_tools()
    assert [t.name for t in runtime.tools] == ["resolve-library-id", "query-docs"]
    call.assert_not_called()


@pytest.mark.asyncio
async def test_context7_uses_current_owner_key_for_each_call(monkeypatch):
    monkeypatch.setattr(credentials, "personal_key", AsyncMock(side_effect=["fake-alice", "fake-bob"]))
    call = AsyncMock(return_value="docs")
    monkeypatch.setattr(context7, "call_personal_tool", call)
    for owner in ("alice", "bob"):
        runtime = SimpleNamespace(context=SimpleNamespace(user_id=owner))
        assert await context7.resolve_library_id.coroutine("python", "question", runtime) == "docs"
    targets = [c.args[0] for c in call.call_args_list]
    assert [t["mcpServers"]["context7"]["headers"]["CONTEXT7_API_KEY"] for t in targets] == ["fake-alice", "fake-bob"]


@pytest.mark.asyncio
async def test_context7_failure_has_no_secret_or_raw_provider_error(monkeypatch):
    monkeypatch.setattr(credentials, "personal_key", AsyncMock(return_value="secret"))
    monkeypatch.setattr(context7, "call_personal_tool", AsyncMock(side_effect=RuntimeError("secret")))
    result = await context7.query_docs.coroutine("/python/docs", "question", SimpleNamespace(context=SimpleNamespace(user_id="alice")))
    assert "secret" not in str(result)
    assert "error" in result
