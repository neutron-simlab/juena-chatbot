"""Deployment settings that protect users' browsers, pinned so they stay set.

None of these is Python the unit suite would otherwise run: they live in nginx,
Streamlit and Compose files, where deleting one line silently removes the
protection and nothing fails.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: What nginx must add to every answer for JüNA, and why.
SECURITY_HEADERS = {
    # Browsers remember to use HTTPS for this host only.
    "Strict-Transport-Security": "max-age=31536000",
    # Never guess a file's type from its contents.
    "X-Content-Type-Options": "nosniff",
    # No other website may show JüNA inside a frame, so nobody can be tricked
    # into clicking "Approve" on a page they cannot see. Both spellings: the
    # first is the current standard, the second is what older browsers read.
    "Content-Security-Policy": "frame-ancestors 'self'",
    "X-Frame-Options": "SAMEORIGIN",
    # The address bar holds the conversation id; do not send it to other sites.
    # `same-origin` rather than `no-referrer`: with the latter the browser
    # labels the sign-out form as coming from nowhere, and it is refused.
    "Referrer-Policy": "same-origin",
}


def _juena_server_block(path: Path) -> str:
    """The `server { … }` block that proxies to JüNA's Streamlit port."""

    blocks = re.split(r"(?m)^server \{", path.read_text(encoding="utf-8"))
    matching = [block for block in blocks if ":9501" in block]
    assert len(matching) == 1, f"expected exactly one JüNA server block in {path.name}"
    return matching[0]


@pytest.mark.parametrize("config", ["deploy/nginx.conf", "deploy/shared-proxy/nginx.with-juena.conf"])
def test_the_gateway_sends_the_browser_security_headers(config: str) -> None:
    block = _juena_server_block(ROOT / config)

    for name, value in SECURITY_HEADERS.items():
        # `always` matters: without it nginx leaves the header off error pages.
        assert re.search(rf'add_header\s+{name}\s+"{re.escape(value)}"\s+always;', block), name
    assert re.search(r"server_tokens\s+off;", block)


@pytest.mark.parametrize("config", ["deploy/nginx.conf", "deploy/shared-proxy/nginx.with-juena.conf"])
def test_no_location_drops_the_inherited_headers(config: str) -> None:
    """nginx inherits `add_header` only into a block that has none of its own."""

    block = _juena_server_block(ROOT / config)

    # A line that *starts* a location block; the word also appears in comments.
    locations = re.findall(r"(?m)^\s*location[^{\n]*\{[^}]*\}", block)
    assert locations, "found no location blocks to check"
    for location in locations:
        assert "add_header" not in location, location


def test_streamlit_keeps_usage_statistics_and_cross_site_protection_as_intended() -> None:
    settings = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    # Otherwise every user's browser reports usage to Streamlit's own servers.
    assert settings["browser"]["gatherUsageStats"] is False
    # Both are Streamlit's defaults. With either off, a page on another host
    # could open the app's live connection with the user's cookie attached.
    assert settings["server"]["enableCORS"] is True
    assert settings["server"]["enableXsrfProtection"] is True


def _compose_service(filename: str) -> dict:
    return yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))["services"]["juena-chatbot"]


def test_production_hides_error_traces_and_development_keeps_them() -> None:
    """A Python traceback in the browser shows file paths and source lines."""

    assert "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none" in _compose_service("docker-compose.yml")["environment"]
    assert "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=full" in _compose_service("docker-compose.dev.yml")["environment"]


def test_the_app_container_cannot_gain_privileges() -> None:
    assert _compose_service("docker-compose.yml")["security_opt"] == ["no-new-privileges:true"]
