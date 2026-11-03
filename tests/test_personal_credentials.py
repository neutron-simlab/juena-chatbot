"""Personal credential ownership, API enforcement, and replacement semantics."""

import asyncio
import dataclasses
import json
import os
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import select

import juena_core.config as core_config
from juena.core.config import global_config
from juena.server import credentials as keys
from juena.server.credential_endpoints import router, require_chat_credentials
from juena.server.auth.endpoints import router as auth_router
from juena.server.database.models import UserAPICredential
from juena_core.server.api.endpoints import build_api_router
from juena_core.server.database.connection import database_lifespan, get_session_factory
from juena_core.server.identity import create_auth_session, revoke_auth_session, session_principal, upsert_principal


def test_ciphertext_is_bound_to_owner_service_and_endpoint(monkeypatch):
    owner = uuid4()
    endpoint = keys.service_endpoint("blablador")
    row = UserAPICredential(user_id=owner, service="blablador", endpoint=endpoint,
                            encrypted_key=keys._encrypt("fake-secret", owner, "blablador", endpoint))
    assert "fake-secret" not in row.encrypted_key
    assert keys._decrypt(row, owner, "blablador") == "fake-secret"
    for other_owner, service in [(uuid4(), "blablador"), (owner, "tavily")]:
        with pytest.raises(keys.CredentialError):
            keys._decrypt(row, other_owner, service)
    monkeypatch.setattr(global_config, "BLABLADOR_BASE_URL", "https://different.invalid/v1")
    with pytest.raises(keys.CredentialError):
        keys._decrypt(row, owner, "blablador")
    monkeypatch.setattr(global_config, "BLABLADOR_BASE_URL", endpoint)
    monkeypatch.setattr(global_config, "JUENA_SECRET_KEY", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=")
    with pytest.raises(keys.CredentialError):
        keys._decrypt(row, owner, "blablador")


@pytest_asyncio.fixture
async def account_api(monkeypatch):
    dsn = os.getenv("JUENA_CREDENTIAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set JUENA_CREDENTIAL_TEST_DATABASE_URL to an isolated PostgreSQL database")
    monkeypatch.setattr(core_config, "_settings", dataclasses.replace(core_config.settings(), DATABASE_URL=dsn))
    monkeypatch.setattr(global_config, "UI_PUBLIC_URL", "http://localhost:9501")
    monkeypatch.setattr(global_config, "JUENA_PUBLIC_URL", "http://localhost:8080")
    async def checked(*args):
        pass
    monkeypatch.setattr(keys, "check_key", checked)
    async with database_lifespan():
        users = []
        async with get_session_factory()() as session:
            for _ in range(2):
                user = await upsert_principal(session, subject=str(uuid4()), issuer="urn:test:credentials")
                token = await create_auth_session(session, user)
                users.append((user, token))
            await session.commit()
        app = FastAPI()
        app.include_router(router)
        app.include_router(auth_router)
        app.include_router(build_api_router(session_principal, invocation_guard=require_chat_credentials))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost:8080",
                                     headers={"Origin": "http://localhost:9501"},
                                     cookies={"juena_session": users[0][1]}) as client:
            yield client, users


def payload(service="blablador", key="fake-alice-key"):
    return {"key": key, "endpoint": keys.service_endpoint(service)}


@pytest.mark.asyncio
async def test_owner_only_status_replace_remove_and_no_secret_responses(account_api, monkeypatch):
    client, ((alice, _), (bob, bob_token)) = account_api
    path = "/auth/me/credentials/blablador"
    assert (await client.get("/auth/me")).json()["chat_ready"] is False
    saved = await client.put(path, json=payload())
    assert saved.status_code == 200
    assert "fake-alice-key" not in saved.text
    assert saved.headers["cache-control"] == "no-store"
    assert await keys.personal_key(str(alice.id), "blablador") == "fake-alice-key"
    bob_status = await client.get("/auth/me", headers={"Cookie": f"juena_session={bob_token}"})
    assert bob_status.json()["chat_ready"] is False
    with pytest.raises(keys.CredentialError):
        await keys.personal_key(str(bob.id), "blablador")
    async with get_session_factory()() as session:
        row = await session.get(UserAPICredential, (alice.id, "blablador"))
        assert "fake-alice-key" not in row.encrypted_key
    async def rejected(*args):
        raise keys.CredentialError("Key rejected", code="credential_invalid", status=400)
    monkeypatch.setattr(keys, "check_key", rejected)
    refused = await client.put(path, json=payload(key="fake-rejected"))
    assert refused.status_code == 400
    assert await keys.personal_key(str(alice.id), "blablador") == "fake-alice-key"
    assert (await client.delete(path, params={"endpoint": keys.service_endpoint("blablador")})).status_code == 200
    with pytest.raises(keys.CredentialError):
        await keys.personal_key(str(alice.id), "blablador")
    assert (await client.get("/auth/me")).json()["chat_ready"] is False


@pytest.mark.asyncio
async def test_validation_origin_and_endpoint_refuse_before_external_call(account_api, monkeypatch):
    client, _ = account_api
    calls = []
    async def checked(*args):
        calls.append(args)
    monkeypatch.setattr(keys, "check_key", checked)
    path = "/auth/me/credentials/blablador"
    for value in ["", "x" * 4097, ["secret"], "line\nbreak"]:
        response = await client.put(path, json=payload(key=value))
        assert response.status_code == 400
        assert "line\\nbreak" not in response.text and "secret" not in response.text
    assert (await client.put(path, json={**payload(), "user_id": str(uuid4())})).status_code == 400
    assert (await client.put(path, json=payload(), headers={"Origin": "https://evil.invalid"})).status_code == 403
    assert (await client.delete(path, params={"endpoint": "https://old.invalid"})).status_code == 409
    assert (await client.put(path, json={**payload(), "endpoint": "https://evil.invalid"})).status_code == 409
    assert (await client.put(path, json=payload(), headers={"Cookie": "juena_session=invalid"})).status_code == 401
    assert calls == []


@pytest.mark.asyncio
async def test_logout_during_check_cannot_save_key(account_api, monkeypatch):
    client, ((alice, token), _) = account_api
    async def revoke_during_check(*args):
        async with get_session_factory()() as session:
            await revoke_auth_session(session, token)
            await session.commit()
    monkeypatch.setattr(keys, "check_key", revoke_during_check)
    response = await client.put("/auth/me/credentials/blablador", json=payload())
    assert response.status_code == 401
    async with get_session_factory()() as session:
        assert await session.get(UserAPICredential, (alice.id, "blablador")) is None


@pytest.mark.asyncio
async def test_readiness_is_enforced_for_all_invocation_routes(account_api):
    client, _ = account_api
    for path, kwargs in [
        ("/stream", {"json": {"message": "hello"}}),
        ("/juena/stream", {"json": {"message": "hello"}}),
        ("/stream_with_files", {"data": {"message": "hello"}}),
        ("/juena/stream_with_files", {"data": {"message": "hello"}}),
        ("/resume", {"json": {"thread_id": "test", "kind": "clarification", "interrupt_id": "test", "answer": "yes"}}),
    ]:
        response = await client.post(path, **kwargs)
        assert response.status_code == 403, (path, response.text)
        assert response.json()["detail"]["code"] == "credential_required"
    response = await client.post("/stream", json={"message": "hello", "provider": "openai"})
    assert response.status_code == 400
    assert (await client.get("/auth/me")).status_code == 200


@pytest.mark.asyncio
async def test_concurrent_requests_and_key_changes_use_current_owner(account_api, monkeypatch):
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.runtime import get_runtime
    from juena.core.personal_models import blablador_key
    from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

    client, ((alice, _), (bob, bob_token)) = account_api
    path = "/auth/me/credentials/blablador"
    await client.put(path, json=payload())
    await client.put(path, json=payload(key="fake-bob-key"), headers={"Cookie": f"juena_session={bob_token}"})
    observed = []
    async def reply(request):
        observed.append((get_runtime().context.user_id, request.headers["authorization"]))
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
            "model": "mock", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as transport:
        model = ChatOpenAI(model="mock", api_key=blablador_key, base_url="https://mock.invalid/v1",
                           http_async_client=transport, max_retries=0)
        graph = create_agent(model=model, context_schema=RuntimeModelContext, checkpointer=InMemorySaver())
        async def run(user):
            config = {"configurable": {"thread_id": str(uuid4())}}
            result = await graph.ainvoke({"messages": [("user", "hello")]}, config,
                context=RuntimeModelContext(provider="blablador", model="mock", user_id=str(user.id)))
            assert "fake-alice-key" not in json.dumps(result, default=str)
        await asyncio.gather(*(run(user) for user in [alice, bob] * 5))
        assert all(key == f"Bearer fake-{'alice' if user == str(alice.id) else 'bob'}-key" for user, key in observed)
        await client.put(path, json=payload(key="fake-replacement"))
        await run(alice)
        assert observed[-1][1] == "Bearer fake-replacement"
        await client.delete(path, params={"endpoint": keys.service_endpoint("blablador")})
        count = len(observed)
        with pytest.raises(keys.CredentialError):
            await run(alice)
        assert len(observed) == count


@pytest.mark.asyncio
async def test_summarization_fallback_and_streaming_use_personal_key(account_api):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelFallbackMiddleware, SummarizationMiddleware
    from langchain_openai import ChatOpenAI
    from langgraph.runtime import get_runtime
    from juena.core.personal_models import blablador_key
    from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

    client, ((alice, _), _) = account_api
    await client.put("/auth/me/credentials/blablador", json=payload())
    seen = []
    async def reply(request):
        body = json.loads(request.content)
        seen.append((body["model"], request.headers["authorization"], get_runtime().context.user_id))
        if body["model"] == "primary":
            return httpx.Response(503, json={"error": {"message": "offline", "type": "server_error"}})
        if body.get("stream"):
            chunk = {"id": "mock", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]}
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "summary"}, "finish_reason": "stop"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as transport:
        def model(name):
            return ChatOpenAI(model=name, api_key=blablador_key, base_url="https://mock.invalid/v1",
                              http_async_client=transport, max_retries=0)
        graph = create_agent(model=model("primary"), context_schema=RuntimeModelContext,
            middleware=[SummarizationMiddleware(model=model("summary"), trigger=("messages", 3), keep=("messages", 1)),
                        ModelFallbackMiddleware(model("fallback"))])
        async for _ in graph.astream({"messages": [("user", "first"), ("assistant", "answer"), ("user", "next"),
                                                   ("assistant", "answer"), ("user", "last")]},
                                     context=RuntimeModelContext(provider="blablador", model="primary", user_id=str(alice.id)),
                                     stream_mode="messages"):
            pass
    assert {entry[0] for entry in seen} == {"primary", "summary", "fallback"}
    assert all(key == "Bearer fake-alice-key" and owner == str(alice.id) for _, key, owner in seen)


def test_provider_failure_messages_omit_raw_response_and_credentials():
    from openai import AuthenticationError, RateLimitError
    from juena.core.personal_models import credential_error_message
    request = httpx.Request("POST", "https://models.invalid/chat/completions")
    for code, error_type, expected in [(401, AuthenticationError, "My profile"), (429, RateLimitError, "rate limiting")]:
        error = error_type("fake-secret-in-upstream-error", response=httpx.Response(code, request=request), body={})
        message = credential_error_message(error)
        assert expected in message
        assert "fake-secret" not in message


@pytest.mark.asyncio
async def test_background_recovery_resolves_job_owners_current_key(account_api, monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from juena.core.personal_models import blablador_key
    from juena.research import runner

    client, ((alice, _), _) = account_api
    path = "/auth/me/credentials/blablador"
    await client.put(path, json=payload())
    seen = []
    async def step(state):
        seen.append(await blablador_key())
        return {"messages": []}
    builder = StateGraph(MessagesState)
    builder.add_node("first", step)
    builder.add_node("second", step)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_after=["first"])
    job = SimpleNamespace(id=uuid4(), user_id=alice.id, thread_id="background-test",
        started_at=datetime.now(UTC), objective="test", run_input={"files": {}, "provider": "blablador", "model": "mock"})
    monkeypatch.setattr(runner, "get_filesystem", lambda: None)
    await runner._invoke(graph, job)
    await client.put(path, json=payload(key="fake-recovered-key"))
    await runner._invoke(graph, job)
    assert seen == ["fake-alice-key", "fake-recovered-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("service,status,expected", [("blablador", 401, 400), ("tavily", 429, 503), ("blablador", 200, None)])
async def test_provider_check_uses_fixed_endpoint_and_redacts_errors(monkeypatch, service, status, expected):
    real_client = httpx.AsyncClient
    requests = []
    async def reply(request):
        requests.append(request)
        return httpx.Response(status, json={"data": [], "error": "fake-secret"})
    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(reply), **kwargs)
    monkeypatch.setattr(keys.httpx, "AsyncClient", client)
    if expected is None:
        await keys.check_key(service, "fake-secret", keys.service_endpoint(service))
    else:
        with pytest.raises(keys.CredentialError) as error:
            await keys.check_key(service, "fake-secret", keys.service_endpoint(service))
        assert error.value.status == expected
        assert "fake-secret" not in str(error.value)
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer fake-secret"
    assert requests[0].url.path.endswith("/models" if service == "blablador" else "/usage")
