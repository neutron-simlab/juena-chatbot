"""Profile forms clear credentials and onboarding gives the exact token scope."""

from unittest.mock import Mock
from streamlit.testing.v1 import AppTest


def _page():
    from app.profile_page import render_profile_page
    render_profile_page()


def _profile():
    page = AppTest.from_function(_page)
    page.session_state.user = {
        "id": "alice", "display_name": None, "email": None, "chat_ready": False,
        "credentials": {service: {"saved": False, "available": False, "validated_at": None,
                         "endpoint": f"https://{service}.invalid"} for service in ("blablador", "tavily", "context7")},
    }
    page.session_state.client = Mock()
    return page.run()


def test_profile_explains_key_scope_and_missing_saml_attributes():
    page = _profile()
    assert not page.exception
    assert any("read_user" in element.value and "blablador_api_access" in element.value for element in page.markdown)
    assert all("Not provided by IFFLogin" in element.value for element in page.text)
    assert len(page.text_input) == 3
    assert all(element.value == "" for element in page.text_input)


def test_submit_clears_secret_field_and_uses_displayed_endpoint():
    page = _profile()
    client = page.session_state.client
    page.text_input[0].set_value("fake-personal-secret")
    page.button[0].click().run()
    assert not page.exception
    client.save_api_key.assert_called_once_with("blablador", "fake-personal-secret", "https://blablador.invalid")
    assert page.text_input[0].value == ""
    assert "fake-personal-secret" not in str([element.value for element in page.markdown])


def test_failed_save_also_clears_secret_field():
    from juena.clients.client import AgentClientError
    page = _profile()
    page.session_state.client.save_api_key.side_effect = AgentClientError("Key rejected")
    page.text_input[0].set_value("fake-rejected-secret")
    page.button[0].click().run()
    assert not page.exception
    assert page.text_input[0].value == ""
    assert page.error[0].value == "Key rejected"
