"""PySAML2 service-provider integration for the FZJ identity provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.client import Saml2Client
from saml2.config import SPConfig
from saml2.metadata import create_metadata_string
from saml2.saml import NameID

from juena.core.config import global_config

PERSISTENT_NAME_ID_FORMAT = "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"


class SamlAuthenticationError(ValueError):
    """Raised when a SAML response cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SamlIdentity:
    subject: str
    subject_format: str
    issuer: str
    in_response_to: str
    email: str | None = None
    display_name: str | None = None


def _first_attribute(attributes: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = attributes.get(name)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value:
            return value
    return None


class SamlService:
    """Small, framework-neutral wrapper around PySAML2."""

    def __init__(self, config: SPConfig) -> None:
        self.config = config
        self.client = Saml2Client(config=config)

    @classmethod
    def from_config(cls) -> "SamlService":
        required = {
            "SAML_SP_ENTITY_ID": global_config.SAML_SP_ENTITY_ID,
            "SAML_SP_ACS_URL": global_config.SAML_SP_ACS_URL,
            "SAML_SP_SLS_URL": global_config.SAML_SP_SLS_URL,
            "SAML_SP_CERT_FILE": global_config.SAML_SP_CERT_FILE,
            "SAML_SP_KEY_FILE": global_config.SAML_SP_KEY_FILE,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required SAML settings: {', '.join(missing)}")
        for setting in ("SAML_SP_CERT_FILE", "SAML_SP_KEY_FILE"):
            path = Path(str(required[setting]))
            if not path.is_file():
                raise RuntimeError(f"{setting} does not exist: {path}")

        settings = {
            "entityid": global_config.SAML_SP_ENTITY_ID,
            "xmlsec_binary": global_config.XMLSEC_BINARY,
            "key_file": global_config.SAML_SP_KEY_FILE,
            "cert_file": global_config.SAML_SP_CERT_FILE,
            "metadata": {
                "remote": [{"url": global_config.SAML_IDP_METADATA_URL}],
            },
            "service": {
                "sp": {
                    "endpoints": {
                        "assertion_consumer_service": [
                            (global_config.SAML_SP_ACS_URL, BINDING_HTTP_POST),
                        ],
                        "single_logout_service": [
                            (global_config.SAML_SP_SLS_URL, BINDING_HTTP_REDIRECT),
                            (global_config.SAML_SP_SLS_URL, BINDING_HTTP_POST),
                        ],
                    },
                    "allow_unsolicited": False,
                    "authn_requests_signed": True,
                    "logout_requests_signed": True,
                    "want_response_signed": True,
                    "want_assertions_signed": True,
                    "name_id_format": PERSISTENT_NAME_ID_FORMAT,
                },
            },
        }
        config = SPConfig()
        config.load(settings)
        return cls(config)

    def prepare_login(self, relay_state: str) -> tuple[str, str]:
        request_id, info = self.client.prepare_for_authenticate(
            relay_state=relay_state,
            binding=BINDING_HTTP_REDIRECT,
            response_binding=BINDING_HTTP_POST,
            nameid_format=PERSISTENT_NAME_ID_FORMAT,
            sign=True,
        )
        headers = dict(info.get("headers", []))
        redirect_url = headers.get("Location")
        if not redirect_url:
            raise RuntimeError("SAML client did not produce a login redirect")
        return str(request_id), str(redirect_url)

    def parse_response(
        self,
        encoded_response: str,
        outstanding: Mapping[str, str],
    ) -> SamlIdentity:
        try:
            response = self.client.parse_authn_request_response(
                encoded_response,
                BINDING_HTTP_POST,
                outstanding=dict(outstanding),
            )
        except Exception as exc:
            raise SamlAuthenticationError("SAML response validation failed") from exc
        if response is None or not response.is_authenticated():
            raise SamlAuthenticationError("SAML response is not authenticated")

        session_info = response.session_info()
        name_id = session_info.get("name_id")
        subject = str(getattr(name_id, "text", "") or name_id or "").strip()
        subject_format = str(getattr(name_id, "format", "") or "")
        in_response_to = str(
            getattr(response, "in_response_to", "")
            or session_info.get("in_response_to", "")
        )
        issuer = str(session_info.get("issuer", ""))
        if not subject or not in_response_to or not issuer:
            raise SamlAuthenticationError("SAML response is missing identity binding fields")
        if subject_format != PERSISTENT_NAME_ID_FORMAT:
            raise SamlAuthenticationError("SAML response did not provide a persistent NameID")

        attributes = session_info.get("ava") or {}
        return SamlIdentity(
            subject=subject,
            subject_format=subject_format,
            issuer=issuer,
            in_response_to=in_response_to,
            email=_first_attribute(
                attributes,
                "mail",
                "email",
                "urn:oid:0.9.2342.19200300.100.1.3",
            ),
            display_name=_first_attribute(
                attributes,
                "displayName",
                "cn",
                "urn:oid:2.16.840.1.113730.3.1.241",
            ),
        )

    def metadata_xml(self) -> bytes:
        return create_metadata_string(None, config=self.config)

    def prepare_logout(self, subject: str) -> str | None:
        """Return an IdP redirect for SP-initiated logout when available."""
        name_id = NameID(text=subject, format=PERSISTENT_NAME_ID_FORMAT)
        responses = self.client.global_logout(name_id, sign=True)
        if not isinstance(responses, dict):
            return None
        for response in responses.values():
            if not isinstance(response, tuple) or len(response) != 2:
                continue
            _binding, http_info = response
            headers = dict(http_info.get("headers", []))
            if location := headers.get("Location"):
                return str(location)
        return None

    def parse_logout_response(self, encoded_response: str, binding: str) -> None:
        """Validate a LogoutResponse returned by the IdP after global logout."""

        try:
            self.client.parse_logout_request_response(encoded_response, binding)
        except Exception as exc:
            raise SamlAuthenticationError("SAML logout response validation failed") from exc

    def parse_logout_request(
        self,
        encoded_request: str,
        binding: str,
        relay_state: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Handle an IdP-initiated LogoutRequest.

        Returns ``(saml_subject, redirect_url)``.  The subject identifies whose
        local sessions must be revoked; the redirect delivers our
        LogoutResponse back to the IdP when it supplied one.
        """

        try:
            request = self.client.parse_logout_request(encoded_request, binding)
        except Exception as exc:
            raise SamlAuthenticationError("SAML logout request validation failed") from exc

        name_id = getattr(request.message, "name_id", None)
        subject = str(getattr(name_id, "text", "") or "").strip() or None

        redirect_url: str | None = None
        try:
            http_info = self.client.handle_logout_request(
                encoded_request,
                name_id,
                binding,
                relay_state=relay_state,
                sign=True,
            )
            headers = dict(http_info.get("headers", []))
            redirect_url = headers.get("Location")
        except Exception:
            # The local revocation below is authoritative; failing to answer
            # the IdP must not leave the session alive.
            redirect_url = None

        return subject, redirect_url


_saml_service: SamlService | None = None


def get_saml_service() -> SamlService:
    if _saml_service is None:
        raise RuntimeError("SAML service is not initialized")
    return _saml_service


@asynccontextmanager
async def saml_lifespan() -> AsyncGenerator[SamlService, None]:
    global _saml_service
    service = SamlService.from_config()
    _saml_service = service
    try:
        yield service
    finally:
        _saml_service = None
