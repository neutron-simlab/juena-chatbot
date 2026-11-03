"""Trust, signature and refresh regressions using a disposable local IdP."""

import asyncio
import base64
import shutil
import subprocess
import time
import zlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.config import IdPConfig
from saml2.httpbase import HTTPBase
from saml2.metadata import create_metadata_string
from saml2.saml import NameID
from saml2.server import Server

from juena.server.auth import saml


@pytest.fixture
def metadata_setup(tmp_path, monkeypatch):
    xmlsec = shutil.which("xmlsec1")
    if not xmlsec:
        pytest.skip("xmlsec1 required; these tests also run in the deployment image")
    key, cert = tmp_path / "idp.key", tmp_path / "idp.crt"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=deployment-test", "-keyout", str(key), "-out", str(cert),
    ], check=True, capture_output=True)
    config = IdPConfig()
    config.load({
        "entityid": "https://idp.example/metadata", "xmlsec_binary": xmlsec,
        "key_file": str(key), "cert_file": str(cert), "valid_for": 24,
        "signing_algorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
        "digest_algorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
        "service": {"idp": {"endpoints": {
            "single_sign_on_service": [("https://idp.example/login", BINDING_HTTP_REDIRECT)],
            "single_logout_service": [("https://idp.example/logout", BINDING_HTTP_REDIRECT),
                                      ("https://idp.example/logout/post", BINDING_HTTP_POST)],
        }}},
    })
    response = SimpleNamespace(status_code=200, content=create_metadata_string(None, config=config, sign=True))
    monkeypatch.setattr(HTTPBase, "send", lambda *a, **kw: response)
    settings = {
        "XMLSEC_BINARY": xmlsec,
        "SAML_SP_ENTITY_ID": "https://sp.example/auth/metadata",
        "SAML_SP_ACS_URL": "https://sp.example/auth/acs",
        "SAML_SP_SLS_URL": "https://sp.example/auth/sls",
        "SAML_SP_CERT_FILE": str(cert), "SAML_SP_KEY_FILE": str(key),
        "SAML_IDP_METADATA_URL": "https://idp.example/metadata",
        "SAML_IDP_METADATA_CERT_FILE": str(cert), "SAML_METADATA_REFRESH_SECONDS": 60,
        "SAML_ORGANIZATION_NAME": "Test institute",
        "SAML_ORGANIZATION_URL": "https://institute.example",
        "SAML_CONTACT_NAME": "Technical contact", "SAML_CONTACT_EMAIL": "admin@example.org",
    }
    for name, value in settings.items():
        monkeypatch.setattr(saml.global_config, name, value)
    return config, response


def test_signed_metadata_contact_and_sha256_login(metadata_setup):
    service = saml.SamlService.from_config()
    _, location = service.prepare_login("relay")
    assert parse_qs(urlparse(location).query)["SigAlg"] == [
        "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
    ]
    assert service.client.digest_algorithm == "http://www.w3.org/2001/04/xmlenc#sha256"
    xml = service.metadata_xml()
    if isinstance(xml, bytes):
        xml = xml.decode()
    assert "admin@example.org" in xml and "Test institute" in xml
    assert 'WantAssertionsSigned="true"' in xml
    assert service.metadata_expires_at > datetime.now(UTC)


@pytest.mark.parametrize("tamper", [False, True])
def test_unsigned_and_tampered_metadata_rejected(metadata_setup, tamper):
    config, response = metadata_setup
    if tamper:
        response.content = response.content.replace("idp.example/login", "evil.example/login")
    else:
        response.content = create_metadata_string(None, config=config, sign=False)
    with pytest.raises(Exception):
        saml.SamlService.from_config()


def test_refresh_preserves_client_and_rejects_expired_metadata(metadata_setup, monkeypatch):
    service = saml.SamlService.from_config()
    client, store = service.client, service.config.metadata
    client.state["pending-logout"] = {"subject": "pilot"}
    replacement = saml.SamlService.from_config()
    service.refresh_metadata(replacement)
    assert service.client is client
    assert service.config.metadata is store
    assert client.metadata is store and client.sec.metadata is store
    assert client.state["pending-logout"] == {"subject": "pilot"}
    monkeypatch.setattr(saml, "_saml_service", service)
    assert saml.get_saml_service() is service
    service.metadata_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(HTTPException) as exc:
        saml.get_saml_service()
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_failed_refresh_keeps_verified_metadata_and_retries(monkeypatch):
    import unittest.mock

    service = SimpleNamespace(metadata_expires_at=None, refresh_metadata=unittest.mock.Mock())
    replacement = object()
    attempts = iter([RuntimeError("metadata unavailable"), replacement])

    def fetch():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps = 0

    async def sleep(delay):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(saml.SamlService, "from_config", fetch)
    monkeypatch.setattr(saml.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await saml._refresh_metadata(service)
    service.refresh_metadata.assert_called_once_with(replacement)


@pytest.mark.parametrize("binding", [BINDING_HTTP_REDIRECT, BINDING_HTTP_POST])
def test_logout_requests_require_valid_transport_or_xml_signature(metadata_setup, binding):
    config, _ = metadata_setup
    service = saml.SamlService.from_config()
    idp = Server(config=config)
    _, request = idp.create_logout_request(
        "https://sp.example/auth/sls", "https://sp.example/auth/metadata",
        name_id=NameID(text="pilot", format=saml.PERSISTENT_NAME_ID_FORMAT),
        sign=binding == BINDING_HTTP_POST,
    )
    if binding == BINDING_HTTP_REDIRECT:
        info = idp.apply_binding(binding, str(request), "https://sp.example/auth/sls", "relay", sign=True)
        params = {key: value[0] for key, value in parse_qs(urlparse(dict(info["headers"])["Location"]).query).items()}
        encoded = params["SAMLRequest"]
        kwargs = {"relay_state": params["RelayState"], "sigalg": params["SigAlg"], "signature": params["Signature"]}
        assert service.parse_logout_request(encoded, binding, **kwargs)[0] == "pilot"
        with pytest.raises(saml.SamlAuthenticationError):
            service.parse_logout_request(encoded, binding)
        kwargs["relay_state"] = "tampered"
        with pytest.raises(saml.SamlAuthenticationError):
            service.parse_logout_request(encoded, binding, **kwargs)
    else:
        encoded = base64.b64encode(str(request).encode()).decode()
        assert service.parse_logout_request(encoded, binding)[0] == "pilot"
        _, unsigned = idp.create_logout_request(
            "https://sp.example/auth/sls", "https://sp.example/auth/metadata",
            name_id=NameID(text="pilot", format=saml.PERSISTENT_NAME_ID_FORMAT), sign=False,
        )
        with pytest.raises(saml.SamlAuthenticationError):
            service.parse_logout_request(base64.b64encode(str(unsigned).encode()).decode(), binding)


def test_post_logout_binding_is_rendered_as_a_browser_form():
    from juena.server.auth.endpoints import _logout_message_response

    message = saml.SamlLogoutMessage.from_http_info({"method": "POST", "data": ["<form>", "</form>"]})
    response = _logout_message_response(message)
    assert response.status_code == 200
    assert response.body == b"<form></form>"
    assert response.headers["cache-control"] == "no-store"


def test_outbound_logout_keeps_name_id_qualifiers_and_uses_sha256(metadata_setup):
    service = saml.SamlService.from_config()
    name = NameID(text="pilot", format=saml.PERSISTENT_NAME_ID_FORMAT,
                  name_qualifier="https://idp.example/metadata", sp_name_qualifier="https://sp.example/auth/metadata")
    service.client.users.add_information_about_person({
        "name_id": name, "issuer": "https://idp.example/metadata", "ava": {},
        "session_index": "idp-session", "not_on_or_after": int(time.time()) + 3600,
    })
    service.config.preferred_binding = {**service.config.preferred_binding,
                                       "single_logout_service": [BINDING_HTTP_REDIRECT]}
    message = service.prepare_logout("pilot")
    params = parse_qs(urlparse(message.redirect_url).query)
    assert params["SigAlg"] == ["http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"]
    xml = zlib.decompress(base64.b64decode(params["SAMLRequest"][0]), -15).decode()
    assert 'NameQualifier="https://idp.example/metadata"' in xml
    assert 'SPNameQualifier="https://sp.example/auth/metadata"' in xml
    assert "idp-session" in xml


@pytest.mark.parametrize("binding", [BINDING_HTTP_REDIRECT, BINDING_HTTP_POST])
def test_logout_response_signature_and_request_correlation(metadata_setup, binding, monkeypatch):
    from unittest.mock import Mock
    from saml2 import samlp

    config, _ = metadata_setup
    service = saml.SamlService.from_config()
    idp = Server(config=config)
    response = idp._status_response(
        samlp.LogoutResponse, idp._issuer(), None, sign=binding == BINDING_HTTP_POST,
        destination="https://sp.example/auth/sls", in_response_to="pending-logout",
    )
    kwargs = {}
    if binding == BINDING_HTTP_REDIRECT:
        info = idp.apply_binding(binding, str(response), "https://sp.example/auth/sls",
                                 "relay", response=True, sign=True)
        params = {key: value[0] for key, value in parse_qs(urlparse(dict(info["headers"])["Location"]).query).items()}
        encoded = params["SAMLResponse"]
        kwargs = {"relay_state": params["RelayState"], "sigalg": params["SigAlg"], "signature": params["Signature"]}
    else:
        encoded = base64.b64encode(str(response).encode()).decode()
    service.client.state["pending-logout"] = {"entity_id": "https://idp.example/metadata"}
    completed = Mock(side_effect=lambda _: service.client.state.pop("pending-logout"))
    monkeypatch.setattr(service.client, "handle_logout_response", completed)
    service.parse_logout_response(encoded, binding, **kwargs)
    completed.assert_called_once()
    with pytest.raises(saml.SamlAuthenticationError):
        service.parse_logout_response(encoded, binding, **kwargs)
