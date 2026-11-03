"""Registration metadata must work before the app and IdP trust are configured."""

import base64
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from saml2.httpbase import HTTPBase

SOURCE = Path(__file__).resolve().parents[1] / "src/juena/server/auth/sp_metadata.py"
spec = importlib.util.spec_from_file_location("offline_sp_metadata", SOURCE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
MD = "{urn:oasis:names:tc:SAML:2.0:metadata}"
DS = "{http://www.w3.org/2000/09/xmldsig#}"


@pytest.fixture
def registration_settings(tmp_path, monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Offline SP export must not fetch IdP metadata")

    monkeypatch.setattr(HTTPBase, "send", reject_network)
    xmlsec = os.environ.get("XMLSEC_BINARY") or shutil.which("xmlsec1")
    if not xmlsec:
        pytest.skip("xmlsec1 is required for real PySAML2 metadata generation")
    key, cert = tmp_path / "sp.key", tmp_path / "sp.crt"
    subprocess.run([
        "openssl", "req", "-new", "-x509", "-newkey", "rsa:2048", "-sha256",
        "-nodes", "-days", "1", "-subj", "/CN=registration-test",
        "-keyout", str(key), "-out", str(cert),
    ], check=True, capture_output=True)
    return exporter.sp_settings(
        entity_id="https://sp.example/auth/metadata",
        acs_url="https://sp.example/auth/acs",
        sls_url="https://sp.example/auth/sls",
        cert_file=str(cert), key_file=str(key), xmlsec_binary=xmlsec,
        organization_name="Example Institute", organization_url="https://institute.example",
        contact_name="Technical Contact", contact_email="admin@example.org",
    )


def test_registration_document_without_idp_or_application_configuration(registration_settings):
    document = exporter.metadata_xml(registration_settings)
    root = ET.fromstring(document)
    assert root.get("entityID") == "https://sp.example/auth/metadata"
    signature = root.find(DS + "Signature")
    assert signature is not None
    assert signature.find(DS + "SignedInfo").find(DS + "SignatureMethod").get("Algorithm") == exporter.SIGNING_ALGORITHM
    reference = signature.find(DS + "SignedInfo").find(DS + "Reference")
    assert reference.get("URI") == "#" + root.get("ID")
    assert reference.find(DS + "DigestMethod").get("Algorithm") == exporter.DIGEST_ALGORITHM
    # A static registration document must not silently expire after a short interval.
    assert root.get("validUntil") is None
    sp = root.find(MD + "SPSSODescriptor")
    assert sp.get("AuthnRequestsSigned") == "true"
    assert sp.get("WantAssertionsSigned") == "false"
    assert sp.findtext(MD + "NameIDFormat") == exporter.PERSISTENT_NAME_ID_FORMAT
    acs = sp.find(MD + "AssertionConsumerService")
    assert acs.get("Location") == "https://sp.example/auth/acs"
    assert acs.get("Binding").endswith(":HTTP-POST")
    assert {node.get("Binding") for node in sp.findall(MD + "SingleLogoutService")} == {
        "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
        "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
    }
    assert root.find(MD + "ContactPerson").findtext(MD + "EmailAddress") == "admin@example.org"
    assert root.find(MD + "Organization").findtext(MD + "OrganizationName") == "Example Institute"
    expected = subprocess.run([
        "openssl", "x509", "-in", registration_settings["cert_file"], "-outform", "DER",
    ], capture_output=True, check=True).stdout
    assert base64.b64decode(sp.find(".//" + DS + "X509Certificate").text) == expected
    assert b"PRIVATE KEY" not in document


def verify_signature(settings, path, certificate=None):
    certificate = certificate or settings["cert_file"]
    return subprocess.run([
        settings["xmlsec_binary"], "--verify", "--pubkey-cert-pem", str(certificate),
        "--trusted-pem", str(certificate),
        "--id-attr:ID", "urn:oasis:names:tc:SAML:2.0:metadata:EntityDescriptor",
        "--enabled-reference-uris", "same-doc", str(path),
    ], capture_output=True)


def test_metadata_signature_verifies_and_rejects_changed_endpoints(registration_settings, tmp_path):
    document = exporter.metadata_xml(registration_settings)
    path = tmp_path / "metadata.xml"
    path.write_bytes(document)
    assert verify_signature(registration_settings, path).returncode == 0
    path.write_bytes(document.replace(b"https://sp.example/auth/acs", b"https://attacker.example/auth/acs"))
    assert verify_signature(registration_settings, path).returncode != 0


def test_metadata_signature_requires_the_known_sp_certificate(registration_settings, tmp_path):
    path = tmp_path / "metadata.xml"
    path.write_bytes(exporter.metadata_xml(registration_settings))
    wrong_key, wrong_cert = tmp_path / "wrong.key", tmp_path / "wrong.crt"
    subprocess.run([
        "openssl", "req", "-new", "-x509", "-newkey", "rsa:2048", "-sha256",
        "-nodes", "-days", "1", "-subj", "/CN=untrusted-signer",
        "-keyout", str(wrong_key), "-out", str(wrong_cert),
    ], check=True, capture_output=True)
    assert verify_signature(registration_settings, path, wrong_cert).returncode != 0


def test_standalone_cli_needs_no_app_environment(registration_settings):
    args = [sys.executable, str(SOURCE)]
    for name, value in {
        "entity-id": registration_settings["entityid"],
        "acs-url": "https://sp.example/auth/acs", "sls-url": "https://sp.example/auth/sls",
        "cert-file": registration_settings["cert_file"], "key-file": registration_settings["key_file"],
        "xmlsec-binary": registration_settings["xmlsec_binary"],
    }.items():
        args += ["--" + name, value]
    # Retain only interpreter/native-library paths, not model credentials or app config.
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "LD_LIBRARY_PATH")}
    result = subprocess.run(args, env=env, capture_output=True, check=True)
    root = ET.fromstring(result.stdout)
    assert root.get("entityID") == "https://sp.example/auth/metadata"
    assert root.find(DS + "Signature") is not None
    assert b"PRIVATE KEY" not in result.stdout


def test_offline_export_rejects_remote_idp_configuration(registration_settings):
    registration_settings["metadata"] = {"remote": [{"url": "https://unregistered-idp.example/metadata"}]}
    with pytest.raises(ValueError, match="without IdP metadata"):
        exporter.metadata_xml(registration_settings)
