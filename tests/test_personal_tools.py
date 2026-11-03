"""One compiled middleware filters capabilities independently for each caller."""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import SystemMessage
from juena.agents import personal_tools
from juena.server.credentials import CredentialError


@pytest.mark.asyncio
async def test_optional_tools_and_prompt_follow_current_user(monkeypatch):
    async def key(user_id, service):
        if user_id == "bob":
            raise CredentialError("No key")
        return "fake-key"
    monkeypatch.setattr(personal_tools, "personal_key", key)
    middleware = personal_tools.PersonalToolsMiddleware("tavily", ("tavily_search",))
    all_tools = [SimpleNamespace(name="tavily_search"), SimpleNamespace(name="read_file")]
    captured = []
    async def handler(request):
        captured.append(request)
    for owner in ("alice", "bob", "alice"):
        request = ModelRequest(model=None, messages=[], system_message=SystemMessage(content="Instructions"),
                               tools=all_tools, runtime=SimpleNamespace(context={"user_id": owner}), state={})
        await middleware.awrap_model_call(request, handler)
    assert [[t.name for t in r.tools] for r in captured] == [
        ["tavily_search", "read_file"], ["read_file"], ["tavily_search", "read_file"]]
    assert "unavailable" in captured[1].system_message.text
    assert "fake-key" not in str(captured)
    assert len(all_tools) == 2
