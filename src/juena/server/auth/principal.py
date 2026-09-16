"""SAML's side of the identity seam.

Core owns what an authenticated caller *is* (:class:`Principal`), where the
session lives, and the cookie dependency that reads it. It deliberately does
not know what a SAML assertion is: ``upsert_principal`` takes the four fields,
not an identity object (01/CP3).

This module is the one place that translates between them, so the SAML
vocabulary stops here and everything downstream speaks core's.
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
    """Record the caller an assertion names, creating the ``users`` row if new."""

    return await upsert_principal(
        session,
        subject=identity.subject,
        issuer=identity.issuer,
        email=identity.email,
        display_name=identity.display_name,
    )
