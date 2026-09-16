from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from juena.server.auth import endpoints as auth_endpoints
from juena.server.auth.saml import (
    PERSISTENT_NAME_ID_FORMAT,
    SamlAuthenticationError,
    SamlIdentity,
    SamlService,
)


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    in_response_to = "request-1"

    def __init__(self, *, name_id_format: str = PERSISTENT_NAME_ID_FORMAT) -> None:
        self.name_id_format = name_id_format

    def is_authenticated(self) -> bool:
        return True

    def session_info(self) -> dict:
        return {
            "name_id": SimpleNamespace(text="persistent-subject", format=self.name_id_format),
            "issuer": "https://ifflogin.fz-juelich.de/saml/idp.xml",
            "ava": {"mail": ["person@example.org"], "displayName": ["Test Person"]},
        }


def _service_with_client(client) -> SamlService:  # noqa: ANN001
    service = object.__new__(SamlService)
    service.client = client
    service.config = object()
    return service


def test_compose_forwards_the_auth_bypass_override_to_the_container() -> None:
    """The plan's one-shot bypass-off command must beat the mounted dotenv.

    Without this Compose entry, ``AUTH_DEV_BYPASS=false ./juena up`` changes
    only the launcher's environment. The application container still reads the
    value from ``/app/.env``, so the documented SAML gate can unknowingly run
    with the development bypass enabled.
    """

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "AUTH_DEV_BYPASS=${AUTH_DEV_BYPASS:-false}" in compose


def test_saml_response_requires_persistent_name_id() -> None:
    client = SimpleNamespace(
        parse_authn_request_response=lambda *args, **kwargs: _Response()
    )
    identity = _service_with_client(client).parse_response("encoded", {"request-1": "/"})

    assert identity.subject == "persistent-subject"
    assert identity.subject_format == PERSISTENT_NAME_ID_FORMAT
    assert identity.in_response_to == "request-1"
    assert identity.email == "person@example.org"

    client.parse_authn_request_response = lambda *args, **kwargs: _Response(
        name_id_format="urn:oasis:names:tc:SAML:2.0:nameid-format:transient"
    )
    with pytest.raises(SamlAuthenticationError, match="persistent NameID"):
        _service_with_client(client).parse_response("encoded", {"request-1": "/"})


def test_saml_validation_errors_are_rejected_without_leaking_details() -> None:
    def invalid_signature(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("detailed xml signature failure")

    service = _service_with_client(
        SimpleNamespace(parse_authn_request_response=invalid_signature)
    )
    with pytest.raises(SamlAuthenticationError, match="validation failed") as exc:
        service.parse_response("tampered", {"request-1": "/"})
    assert "detailed" not in str(exc.value)


@pytest.mark.parametrize(
    "candidate",
    [
        "https://attacker.example/steal",
        "javascript:alert(1)",
        "http://localhost:9501.attacker.example/",
    ],
)
def test_relay_state_rejects_external_destinations(candidate: str) -> None:
    with pytest.raises(HTTPException, match="Unsafe"):
        auth_endpoints._safe_return_url(candidate)


class _Scalars:
    def __init__(self, records: list) -> None:
        self.records = records

    def scalars(self):
        return iter(record for record in self.records if record.consumed_at is None)

    def scalar_one_or_none(self):
        return next(
            (record for record in self.records if record.consumed_at is None),
            None,
        )


class _Session:
    def __init__(self, records: list) -> None:
        self.records = records

    async def execute(self, statement):  # noqa: ANN001
        return _Scalars(self.records)

    async def commit(self) -> None:
        return None


@pytest.mark.asyncio
async def test_acs_consumes_request_once_and_sets_hardened_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SimpleNamespace(
        request_id="request-1",
        relay_state="http://localhost:9501/",
        consumed_at=None,
    )
    session = _Session([record])
    identity = SamlIdentity(
        subject="subject",
        subject_format=PERSISTENT_NAME_ID_FORMAT,
        issuer="https://ifflogin.fz-juelich.de/saml/idp.xml",
        in_response_to="request-1",
    )
    service = SimpleNamespace(parse_response=lambda *args, **kwargs: identity)

    async def fake_upsert(*args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(id="user-1")

    async def fake_session(*args, **kwargs):  # noqa: ANN002, ANN003
        return "opaque-token"

    monkeypatch.setattr(auth_endpoints, "get_saml_service", lambda: service)
    monkeypatch.setattr(auth_endpoints, "upsert_saml_user", fake_upsert)
    monkeypatch.setattr(auth_endpoints, "create_auth_session", fake_session)
    monkeypatch.setattr(auth_endpoints.global_config, "SESSION_COOKIE_SECURE", True)

    response = await auth_endpoints.assertion_consumer_service(
        saml_response="signed-response",
        relay_state="http://localhost:9501/",
        session=session,  # type: ignore[arg-type]
    )

    cookie = response.headers["set-cookie"]
    assert "juena_session=opaque-token" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie
    assert "Max-Age=28800" in cookie
    assert record.consumed_at is not None

    with pytest.raises(HTTPException) as replay:
        await auth_endpoints.assertion_consumer_service(
            saml_response="signed-response",
            relay_state="http://localhost:9501/",
            session=session,  # type: ignore[arg-type]
        )
    assert replay.value.status_code == 401


# ---------------------------------------------------------------------------
# Development authentication bypass
# ---------------------------------------------------------------------------


def _reload_config(monkeypatch, **env: str):
    """Re-import the config module with the given environment."""

    import importlib

    import juena.core.config as config_module

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_module)


def test_dev_bypass_is_refused_outside_development(monkeypatch) -> None:
    """Production must fail to start rather than expose an unauthenticated login."""

    with pytest.raises(RuntimeError, match="AUTH_DEV_BYPASS cannot be enabled"):
        _reload_config(monkeypatch, ENVIRONMENT="production", AUTH_DEV_BYPASS="true")

    # Leave the imported module in its normal development state for other tests.
    _reload_config(monkeypatch, ENVIRONMENT="development", AUTH_DEV_BYPASS="false")


def test_dev_bypass_defaults_to_off(monkeypatch, tmp_path) -> None:
    """Unset means off. Point at an empty dotenv so the developer's own
    .env - which may legitimately enable the bypass - cannot affect this."""

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    monkeypatch.delenv("AUTH_DEV_BYPASS", raising=False)
    config_module = _reload_config(
        monkeypatch,
        JUENA_ENV_PATH=str(empty_env),
        ENVIRONMENT="development",
    )

    assert config_module.global_config.AUTH_DEV_BYPASS is False


@pytest.mark.parametrize("value", ["false", "0", "yes", "on", "", "TRUE-ish"])
def test_dev_bypass_only_accepts_the_exact_string_true(monkeypatch, tmp_path, value) -> None:
    """A near-miss value must fail closed, not accidentally enable the bypass."""

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    config_module = _reload_config(
        monkeypatch,
        JUENA_ENV_PATH=str(empty_env),
        ENVIRONMENT="development",
        AUTH_DEV_BYPASS=value,
    )

    assert config_module.global_config.AUTH_DEV_BYPASS is False


@pytest.mark.asyncio
async def test_dev_login_returns_404_when_bypass_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(auth_endpoints.global_config, "AUTH_DEV_BYPASS", False)

    with pytest.raises(HTTPException) as exc:
        await auth_endpoints.dev_login(return_to=None, session=object())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_saml_login_is_unavailable_while_bypass_is_active(monkeypatch) -> None:
    monkeypatch.setattr(auth_endpoints.global_config, "AUTH_DEV_BYPASS", True)

    with pytest.raises(HTTPException) as exc:
        await auth_endpoints.login(return_to=None, session=object())

    assert exc.value.status_code == 503
    assert "dev-login" in exc.value.detail


@pytest.mark.asyncio
async def test_dev_login_issues_a_real_session_cookie(monkeypatch) -> None:
    """The bypass exercises the production session path, only skipping the IdP."""

    captured: dict = {}

    class _Session:
        async def commit(self) -> None:
            captured["committed"] = True

    async def fake_upsert(session, identity):  # noqa: ANN001
        captured["identity"] = identity
        return SimpleNamespace(id="user-uuid")

    async def fake_create_session(session, user):  # noqa: ANN001
        captured["user"] = user
        return "opaque-dev-token"

    monkeypatch.setattr(auth_endpoints.global_config, "AUTH_DEV_BYPASS", True)
    monkeypatch.setattr(auth_endpoints, "upsert_saml_user", fake_upsert)
    monkeypatch.setattr(auth_endpoints, "create_auth_session", fake_create_session)

    response = await auth_endpoints.dev_login(return_to=None, session=_Session())

    assert response.status_code == 303
    assert captured["committed"] is True
    assert captured["identity"].subject == auth_endpoints.DEV_BYPASS_SUBJECT
    assert captured["identity"].issuer == auth_endpoints.DEV_BYPASS_IDP_ENTITY_ID
    # A persistent identity, so the dev user is stable across restarts.
    assert captured["identity"].subject_format == PERSISTENT_NAME_ID_FORMAT

    cookie = response.headers["set-cookie"]
    assert "juena_session=opaque-dev-token" in cookie
    assert "HttpOnly" in cookie


@pytest.mark.asyncio
async def test_logout_without_idp_redirect_returns_browser_to_ui(monkeypatch) -> None:
    captured: dict = {}

    class _Session:
        async def commit(self) -> None:
            captured["committed"] = True

    class _SamlService:
        def prepare_logout(self, subject: str) -> None:
            captured["subject"] = subject
            return None

    async def fake_revoke(session, token: str) -> None:  # noqa: ANN001
        captured["session"] = session
        captured["token"] = token

    session = _Session()
    monkeypatch.setattr(auth_endpoints, "revoke_auth_session", fake_revoke)
    monkeypatch.setattr(auth_endpoints, "get_saml_service", lambda: _SamlService())
    monkeypatch.setattr(
        auth_endpoints.global_config,
        "UI_PUBLIC_URL",
        "http://localhost:9501",
    )

    response = await auth_endpoints.logout(
        request=SimpleNamespace(cookies={"juena_session": "opaque-token"}),
        user=SimpleNamespace(subject="subject-1"),
        session=session,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "http://localhost:9501"
    assert captured == {
        "session": session,
        "token": "opaque-token",
        "committed": True,
        "subject": "subject-1",
    }
    cookie = response.headers["set-cookie"]
    assert "juena_session=" in cookie
    assert "Max-Age=0" in cookie


@pytest.mark.asyncio
async def test_sp_metadata_is_unavailable_while_bypass_is_active(monkeypatch) -> None:
    """503, not 500.

    `/auth/metadata` is the document handed to the IdP administrator, so the
    first thing anyone tries. Under the bypass there is no SP to describe, and
    an opaque 500 reads like a crash rather than a configuration choice.
    """

    monkeypatch.setattr(auth_endpoints.global_config, "AUTH_DEV_BYPASS", True)

    with pytest.raises(HTTPException) as exc:
        await auth_endpoints.service_provider_metadata()

    assert exc.value.status_code == 503
    assert "AUTH_DEV_BYPASS" in exc.value.detail


@pytest.mark.asyncio
async def test_single_logout_is_unavailable_while_bypass_is_active(monkeypatch) -> None:
    monkeypatch.setattr(auth_endpoints.global_config, "AUTH_DEV_BYPASS", True)

    request = SimpleNamespace(method="GET", query_params={"SAMLRequest": "ignored"})

    with pytest.raises(HTTPException) as exc:
        await auth_endpoints.single_logout_service(request=request, session=object())

    assert exc.value.status_code == 503
