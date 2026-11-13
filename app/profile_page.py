"""SAML identity and the signed-in user's personal service keys."""

import streamlit as st
from juena.clients.client import AgentAuthenticationError, AgentClientError

BLABLADOR_GUIDE = "https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/"
BLABLADOR_TOKENS = "https://codebase.helmholtz.cloud/-/user_settings/personal_access_tokens"


def render_blablador_guidance() -> None:
    """Explain how to create a personal Blablador key and where to save it."""
    st.info("JüNA needs your personal Blablador API key before you can chat. Signing in with IFFLogin does not create this key.")
    st.markdown(
        "1. Sign in or register at [Helmholtz Codebase](https://codebase.helmholtz.cloud) "
        "with your institutional/EduGAIN account.\n"
        f"2. Open [Access Tokens]({BLABLADOR_TOKENS}) and create a personal access token "
        "with the **read_user** scope.\n"
        "3. Copy the token into the Blablador field in **My profile**, then choose **Check and save**.\n\n"
        f"Tokens expire after at most one year; replace yours when it expires. [Full setup guide]({BLABLADOR_GUIDE})."
    )


def open_profile() -> None:
    """Switch the page to My profile; used as a button callback."""
    st.session_state.current_page = "profile"


def render_profile_page() -> None:
    """Render My profile: the user's identity and one key form per service."""
    user = st.session_state.user
    st.title("My profile")
    st.text(f"Name: {user.get('display_name') or 'Not provided by IFFLogin'}")
    st.text(f"Email: {user.get('email') or 'Not provided by IFFLogin'}")
    st.caption("These details come from your sign-in provider.")
    if notice := st.session_state.pop("profile_notice", None):
        st.success(notice)
    if error := st.session_state.pop("profile_error", None):
        st.error(error)
    render_blablador_guidance()
    st.caption("Keys are stored encrypted. Removing a saved key deletes JüNA's copy; revoke the token at its provider to invalidate it there.")
    labels = {"blablador": "Blablador · required for chat", "tavily": "Tavily · optional web search", "context7": "Context7 · optional documentation"}
    for service, label in labels.items():
        status = user["credentials"][service]
        endpoint = status["endpoint"]
        st.subheader(label)
        st.caption(f"Service: {endpoint}")
        if status["available"]:
            st.success(f"Key saved. Last checked: {status['validated_at']}")
        elif status["saved"]:
            st.warning("The saved key cannot be used. Check and save a replacement.")
        else:
            st.caption("No key saved.")
        generation_key = f"key_form_generation_{service}"
        generation = st.session_state.get(generation_key, 0)
        form_id = f"api_key_{user['id']}_{service}_{endpoint}_{generation}"
        with st.form(form_id, clear_on_submit=True):
            key = st.text_input("API key", type="password", autocomplete="off", max_chars=4096)
            submitted = st.form_submit_button("Check and save")
        if submitted:
            # A fresh widget also clears the value on failed submissions and
            # immediate reruns, where clear_on_submit alone is insufficient.
            st.session_state[generation_key] = generation + 1
            try:
                st.session_state.client.save_api_key(service, key, endpoint)
            except AgentAuthenticationError:
                st.rerun()
            except AgentClientError as error:
                st.session_state.profile_error = str(error)
            else:
                st.session_state.profile_notice = f"Your {service.capitalize()} key was checked and saved."
            st.rerun()
        if status["saved"] and st.button("Remove saved key", key=f"remove_{form_id}"):
            try:
                st.session_state.client.remove_api_key(service, endpoint)
            except AgentAuthenticationError:
                st.rerun()
            except AgentClientError as error:
                st.error(str(error))
            else:
                st.session_state.profile_notice = f"Your saved {service.capitalize()} key was removed."
                st.rerun()
