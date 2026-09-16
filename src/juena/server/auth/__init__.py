"""Authentication, session, and SAML integration.

The session store and the `Principal` are core's; the SAML service provider and
the translation into a principal are this application's. See `auth/principal.py`.
"""

from juena.server.auth.principal import (
    SESSION_COOKIE_NAME,
    Principal,
    create_auth_session,
    hash_session_token,
    revoke_auth_session,
    revoke_sessions_for_subject,
    session_principal,
    upsert_saml_user,
)

__all__ = [
    "SESSION_COOKIE_NAME",
    "Principal",
    "create_auth_session",
    "hash_session_token",
    "revoke_auth_session",
    "revoke_sessions_for_subject",
    "session_principal",
    "upsert_saml_user",
]
