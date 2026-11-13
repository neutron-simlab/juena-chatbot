"""Turn a SAML sign-in into juena-core's ``Principal``.

Core owns what a signed-in caller is (:class:`Principal`), where the session
lives, and the cookie dependency that reads it; it knows nothing about SAML.
This module is the one place that translates between the two, so code past it
never deals with SAML terms.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.server.identity import (
    SESSION_COOKIE_NAME,
    Principal,
    create_auth_session,
    hash_session_token,
    revoke_auth_session,
    revoke_sessions_for_subject,
    session_principal,
    upsert_principal,
)

from juena.server.auth.saml import SamlIdentity

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


async def upsert_saml_user(session: AsyncSession, identity: SamlIdentity) -> Principal:
    """Record the caller an assertion names, creating the ``users`` row if new.

    Args:
        session: Database session.
        identity: The verified SAML identity.

    Returns:
        The user, as core's ``Principal``.
    """

    return await upsert_principal(
        session,
        subject=identity.subject,
        issuer=identity.issuer,
        email=identity.email,
        display_name=identity.display_name,
    )
