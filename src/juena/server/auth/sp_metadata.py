"""SP settings and offline metadata export, without loading application or IdP state.

This file can also run directly inside the deployment image during registration:
``python sp_metadata.py --help``. Only public metadata is written to stdout.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from saml2.config import SPConfig
from saml2.metadata import create_metadata_string

PERSISTENT_NAME_ID_FORMAT = "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
SIGNING_ALGORITHM = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
DIGEST_ALGORITHM = "http://www.w3.org/2001/04/xmlenc#sha256"


def sp_settings(
    *,
    entity_id: str,
    acs_url: str,
    sls_url: str,
    cert_file: str,
    key_file: str,
    xmlsec_binary: str,
    organization_name: str = "",
    organization_url: str = "",
    contact_name: str = "",
    contact_email: str = "",
) -> dict[str, Any]:
    """Share the same SP identity, bindings and signing policy with the runtime."""
    settings: dict[str, Any] = {
        "entityid": entity_id,
        "xmlsec_binary": xmlsec_binary,
        "key_file": key_file,
        "cert_file": cert_file,
        "verify_ssl_cert": True,
        "http_client_timeout": 10,
        "service": {
            "sp": {
                "signing_algorithm": SIGNING_ALGORITHM,
                "digest_algorithm": DIGEST_ALGORITHM,
                "endpoints": {
                    "assertion_consumer_service": [(acs_url, BINDING_HTTP_POST)],
                    "single_logout_service": [
                        (sls_url, BINDING_HTTP_REDIRECT),
                        (sls_url, BINDING_HTTP_POST),
                    ],
                },
                "allow_unsolicited": False,
                "authn_requests_signed": True,
                "logout_requests_signed": True,
                # IFFLogin signs the whole Response, covering its assertions.
                # Require that signature; an additional Assertion signature
                # is optional (and is still verified whenever present).
                "want_response_signed": True,
                "want_assertions_signed": False,
                "name_id_format": PERSISTENT_NAME_ID_FORMAT,
            },
        },
    }
    if organization_name:
        settings["organization"] = {
            "name": [(organization_name, "en")],
            "display_name": [(organization_name, "en")],
            "url": [(organization_url, "en")],
        }
    if contact_email:
        settings["contact_person"] = [{
            "given_name": contact_name,
            "email_address": [contact_email],
            "contact_type": "technical",
        }]
    return settings


def metadata_xml(settings: dict[str, Any]) -> bytes:
    """Export an SP document without fetching or trusting any IdP metadata."""
    if "metadata" in settings:
        raise ValueError("Offline export accepts SP settings only, without IdP metadata")
    config = SPConfig()
    config.load(settings)
    return render_metadata_xml(config)


def render_metadata_xml(config: SPConfig) -> bytes:
    """Sign public metadata with the SP key, preserving the resulting XML bytes."""
    document = create_metadata_string(
        None,
        config=config,
        sign=True,
        sign_alg=SIGNING_ALGORITHM,
        digest_alg=DIGEST_ALGORITHM,
    )
    # Pretty-printing after signing would change signed whitespace and break
    # signature verification. Browsers can display the XML tree directly.
    return document.encode("utf-8") if isinstance(document, str) else document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("entity-id", "acs-url", "sls-url", "cert-file", "key-file"):
        parser.add_argument("--" + argument, required=True)
    parser.add_argument("--xmlsec-binary", default="/usr/bin/xmlsec1")
    for argument in ("organization-name", "organization-url", "contact-name", "contact-email"):
        parser.add_argument("--" + argument, default="")
    sys.stdout.buffer.write(metadata_xml(sp_settings(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
