"""The IdP can read SLO callbacks without gaining access to the application API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit

import httpx
import pytest
import pytest_asyncio

from juena.core.config import global_config
from juena.server.auth import endpoints
from juena.server.auth.saml import SamlAuthenticationError, SamlLogoutMessage
from juena_core.server.database.connection import get_db_session
from juena_core.server.identity import session_principal


@pytest_asyncio.fixture
async def slo_browser(monkeypatch):
    from juena.server.service import app

    parsed = urlsplit(global_config.SAML_IDP_METADATA_URL)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    session = SimpleNamespace(commit=AsyncMock())
    service = Mock()
    service.parse_logout_request.return_value = (
        "test-subject", SamlLogoutMessage(redirect_url=origin + "/saml/slo/redirect?SAMLResponse=test"),
    )
    monkeypatch.setattr(endpoints, "_require_saml_service", lambda: service)
    revoke = AsyncMock()
    monkeypatch.setattr(endpoints, "revoke_sessions_for_subject", revoke)
    monkeypatch.setitem(app.dependency_overrides, get_db_session, lambda: session)

    async def unauthenticated():
        from fastapi import HTTPException
        raise HTTPException(401, "Authentication required")

    monkeypatch.setitem(app.dependency_overrides, session_principal, unauthenticated)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="https://sp.example", headers={"Origin": origin}) as client:
        yield SimpleNamespace(client=client, origin=origin, service=service,
                              session=session, revoke=revoke)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_idp_can_read_logout_callback_without_app_cookie(slo_browser, method):
    browser = slo_browser
    payload = {"SAMLRequest": "validated-by-service", "RelayState": "idp-state"}
    response = await browser.client.request(method, "/auth/sls",
        **({"params": payload} if method == "GET" else {"data": payload}))
    assert response.status_code == 303
    assert response.headers["location"].startswith(browser.origin + "/saml/slo/")
    assert response.headers["access-control-allow-origin"] == browser.origin
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "origin" in response.headers["vary"].lower()
    browser.revoke.assert_awaited_once_with(browser.session, "test-subject")
    browser.session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_idp_logout_preflight_is_allowed(slo_browser, method):
    response = await slo_browser.client.options("/auth/sls", headers={
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == slo_browser.origin
    slo_browser.service.parse_logout_request.assert_not_called()
    slo_browser.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_cors_does_not_bypass_logout_signature_validation(slo_browser):
    slo_browser.service.parse_logout_request.side_effect = SamlAuthenticationError("Invalid logout signature")
    response = await slo_browser.client.get("/auth/sls", params={"SAMLRequest": "invalid"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == slo_browser.origin
    slo_browser.revoke.assert_not_awaited()
    slo_browser.session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/auth/me", "/auth/me/credentials/blablador", "/chats", "/health"])
async def test_idp_does_not_gain_cors_access_to_other_routes(slo_browser, path):
    response = await slo_browser.client.get(path)
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["https://untrusted.example", "null"])
async def test_untrusted_origin_cannot_read_logout_response(slo_browser, origin):
    response = await slo_browser.client.get("/auth/sls", params={"SAMLRequest": "valid"},
                                            headers={"Origin": origin})
    assert "access-control-allow-origin" not in response.headers
    response = await slo_browser.client.options("/auth/sls", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
