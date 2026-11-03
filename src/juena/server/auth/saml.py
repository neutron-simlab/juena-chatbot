"""PySAML2 service-provider integration for the FZJ identity provider."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.client import Saml2Client
from saml2.config import SPConfig
from saml2.metadata import create_metadata_string
from saml2.sigver import verify_redirect_signature

from juena.core.config import global_config

PERSISTENT_NAME_ID_FORMAT = "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class SamlLogoutMessage:
    redirect_url: str | None = None
    html: str | None = None

    @classmethod
    def from_http_info(cls, info: Mapping[str, Any]) -> "SamlLogoutMessage | None":
        if location := dict(info.get("headers", [])).get("Location"):
            return cls(redirect_url=str(location))
        if info.get("method") == "POST" and (data := info.get("data")):
            return cls(html="".join(data) if isinstance(data, list) else str(data))
        return None


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
        self.metadata_expires_at = _metadata_expiry(config)

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

        if global_config.SAML_METADATA_REFRESH_SECONDS < 60:
            raise RuntimeError("SAML_METADATA_REFRESH_SECONDS must be at least 60")
        remote_metadata = {"url": global_config.SAML_IDP_METADATA_URL}
        if metadata_cert := global_config.SAML_IDP_METADATA_CERT_FILE:
            if not Path(metadata_cert).is_file():
                raise RuntimeError("SAML_IDP_METADATA_CERT_FILE does not exist")
            remote_metadata["cert"] = metadata_cert

        settings = {
            "entityid": global_config.SAML_SP_ENTITY_ID,
            "xmlsec_binary": global_config.XMLSEC_BINARY,
            "key_file": global_config.SAML_SP_KEY_FILE,
            "cert_file": global_config.SAML_SP_CERT_FILE,
            "verify_ssl_cert": True,
            "http_client_timeout": 10,
            "metadata": {"remote": [remote_metadata]},
            "service": {
                "sp": {
                    "signing_algorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
                    "digest_algorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
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
        if global_config.SAML_ORGANIZATION_NAME:
            settings["organization"] = {
                "name": [(global_config.SAML_ORGANIZATION_NAME, "en")],
                "display_name": [(global_config.SAML_ORGANIZATION_NAME, "en")],
                "url": [(global_config.SAML_ORGANIZATION_URL, "en")],
            }
        if global_config.SAML_CONTACT_EMAIL:
            settings["contact_person"] = [{
                "given_name": global_config.SAML_CONTACT_NAME,
                "email_address": [global_config.SAML_CONTACT_EMAIL],
                "contact_type": "technical",
            }]
        config = SPConfig()
        config.load(settings)
        for source in config.metadata.metadata.values():
            if not source.entity:
                raise RuntimeError("IdP metadata contains no valid entities")
            # PySAML2 verifies a signature when cert is configured, but accepts
            # unsigned metadata too. A pinned trust certificate requires a signature.
            if metadata_cert and not source.signed():
                raise RuntimeError("IdP metadata must be signed by the configured certificate")
        return cls(config)

    def refresh_metadata(self, refreshed: "SamlService") -> None:
        # Keep the existing client and its login/logout session cache. Every
        # client/security reference retains the same MetadataStore object.
        self.config.metadata.metadata = refreshed.config.metadata.metadata
        self.config.metadata.to_old = refreshed.config.metadata.to_old
        self.metadata_expires_at = refreshed.metadata_expires_at

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

    def prepare_logout(self, subject: str) -> SamlLogoutMessage | None:
        """Return the Redirect or POST message for SP-initiated logout."""
        # Preserve the IdP/SP qualifiers and session index cached at login.
        # Reconstructing a NameID from text alone misses that cache entry.
        name_id = next((name for name in self.client.users.subjects()
                        if name.text == subject and name.format == PERSISTENT_NAME_ID_FORMAT), None)
        if name_id is None:
            return None
        responses = self.client.global_logout(name_id, sign=True)
        if not isinstance(responses, dict):
            return None
        for response in responses.values():
            if not isinstance(response, tuple) or len(response) != 2:
                continue
            _binding, http_info = response
            if message := SamlLogoutMessage.from_http_info(http_info):
                return message
        return None

    def _verify_logout_signature(
        self, message, encoded: str, binding: str, parameter: str,
        relay_state: str | None, sigalg: str | None, signature: str | None,
    ) -> None:
        if binding == BINDING_HTTP_POST:
            # The parser verifies XML signatures when present, but does not
            # require them for LogoutRequest/LogoutResponse by default.
            if message is None or message.signature is None:
                raise SamlAuthenticationError("Unsigned SAML logout message")
            return
        if binding != BINDING_HTTP_REDIRECT or not sigalg or not signature:
            raise SamlAuthenticationError("Missing SAML logout signature")
        params = {parameter: encoded, "SigAlg": sigalg, "Signature": signature}
        if relay_state is not None:
            params["RelayState"] = relay_state
        certs = self.config.metadata.certs(message.issuer.text, "idpsso", "signing")
        if not any(verify_redirect_signature(params, self.client.sec.sec_backend, cert)
                   for _, cert in certs):
            raise SamlAuthenticationError("Invalid SAML logout signature")

    def parse_logout_response(
        self, encoded_response: str, binding: str, *, relay_state: str | None = None,
        sigalg: str | None = None, signature: str | None = None,
    ) -> None:
        """Validate a LogoutResponse returned by the IdP after global logout."""

        try:
            response = self.client.parse_logout_request_response(encoded_response, binding)
            self._verify_logout_signature(response.response, encoded_response, binding,
                                          "SAMLResponse", relay_state, sigalg, signature)
            pending = self.client.state.get(response.in_response_to)
            if not pending or pending.get("entity_id") != response.issuer():
                raise SamlAuthenticationError("Unsolicited SAML logout response")
            self.client.handle_logout_response(response)
        except Exception as exc:
            raise SamlAuthenticationError("SAML logout response validation failed") from exc

    def parse_logout_request(
        self,
        encoded_request: str,
        binding: str,
        relay_state: str | None = None,
        sigalg: str | None = None,
        signature: str | None = None,
    ) -> tuple[str | None, SamlLogoutMessage | None]:
        """Handle an IdP-initiated LogoutRequest.

        Returns ``(saml_subject, message)``. The subject identifies whose
        local sessions must be revoked; the message delivers our
        LogoutResponse back to the IdP when it supplied one.
        """

        try:
            request = self.client.parse_logout_request(encoded_request, binding)
            self._verify_logout_signature(request.message, encoded_request, binding,
                                          "SAMLRequest", relay_state, sigalg, signature)
            if not request.verify():
                raise SamlAuthenticationError("Invalid SAML logout request")
        except Exception as exc:
            raise SamlAuthenticationError("SAML logout request validation failed") from exc

        name_id = getattr(request.message, "name_id", None)
        subject = str(getattr(name_id, "text", "") or "").strip() or None

        message: SamlLogoutMessage | None = None
        try:
            http_info = self.client.handle_logout_request(
                encoded_request,
                name_id,
                binding,
                relay_state=relay_state,
                sigalg=sigalg,
                signature=signature,
                sign=True,
            )
            message = SamlLogoutMessage.from_http_info(http_info)
        except Exception:
            # The local revocation below is authoritative; failing to answer
            # the IdP must not leave the session alive.
            logger.warning("Could not deliver SAML logout response", exc_info=True)

        return subject, message


_saml_service: SamlService | None = None


def _metadata_expiry(config: SPConfig) -> datetime | None:
    expiries = []
    for source in config.metadata.metadata.values():
        root = source.entities_descr or source.entity_descr
        descriptors = [root, *getattr(root, "entity_descriptor", [])]
        for descriptor in descriptors:
            if value := getattr(descriptor, "valid_until", None):
                expiries.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
    expiry = min(expiries) if expiries else None
    if expiry is not None and expiry <= datetime.now(UTC):
        raise RuntimeError("IdP metadata has expired")
    return expiry


def get_saml_service() -> SamlService:
    if _saml_service is None:
        raise RuntimeError("SAML service is not initialized")
    expiry = _saml_service.metadata_expires_at
    if expiry is not None and expiry <= datetime.now(UTC):
        raise HTTPException(status_code=503, detail="Identity provider metadata has expired")
    return _saml_service


async def _refresh_metadata(service: SamlService) -> None:
    while True:
        delay = global_config.SAML_METADATA_REFRESH_SECONDS
        if service.metadata_expires_at is not None:
            remaining = (service.metadata_expires_at - datetime.now(UTC)).total_seconds()
            delay = min(delay, max(60, remaining / 2))
        await asyncio.sleep(delay)
        try:
            refreshed = await asyncio.to_thread(SamlService.from_config)
        except Exception:
            # Keep previously verified metadata only for its remaining validity.
            # get_saml_service refuses it after expiry.
            logger.exception("Could not refresh IdP metadata")
        else:
            service.refresh_metadata(refreshed)


@asynccontextmanager
async def saml_lifespan() -> AsyncGenerator[SamlService, None]:
    global _saml_service
    service = await asyncio.to_thread(SamlService.from_config)
    _saml_service = service
    refresher = asyncio.create_task(_refresh_metadata(service), name="saml-metadata-refresh")
    try:
        yield service
    finally:
        refresher.cancel()
        await asyncio.gather(refresher, return_exceptions=True)
        _saml_service = None
