"""Authenticated user/session services and FastAPI dependency."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena.server.database import get_db_session
from juena.server.models import AuthSession, User, utc_now
from juena.server.saml_service import SamlIdentity

SESSION_COOKIE_NAME = "juena_session"


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: UUID
    saml_subject: str
    email: str | None
    display_name: str | None


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def upsert_saml_user(session: AsyncSession, identity: SamlIdentity) -> User:
    result = await session.execute(
        select(User).where(
            User.idp_entity_id == identity.issuer,
            User.saml_subject == identity.subject,
        )
    )
    user = result.scalar_one_or_none()
    now = utc_now()
    if user is None:
        user = User(
            idp_entity_id=identity.issuer,
            saml_subject=identity.subject,
            email=identity.email,
            display_name=identity.display_name,
            last_login_at=now,
        )
        session.add(user)
    else:
        user.email = identity.email
        user.display_name = identity.display_name
        user.last_login_at = now
        user.updated_at = now
    await session.flush()
    return user


async def create_auth_session(session: AsyncSession, user: User) -> str:
    token = secrets.token_urlsafe(32)
    now = utc_now()
    session.add(
        AuthSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=now,
            last_seen_at=now,
            expires_at=now + timedelta(hours=global_config.SESSION_TTL_HOURS),
        )
    )
    await session.flush()
    return token


async def revoke_sessions_for_subject(session: AsyncSession, saml_subject: str) -> int:
    """Revoke every live session for a SAML subject.

    Used by IdP-initiated single logout, where we are told who to log out but
    never see their cookie.
    """

    now = utc_now()
    result = await session.execute(
        select(AuthSession)
        .join(User, User.id == AuthSession.user_id)
        .where(
            User.saml_subject == saml_subject,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
        )
    )
    revoked = 0
    for auth_session in result.scalars():
        auth_session.revoked_at = now
        revoked += 1
    if revoked:
        await session.flush()
    return revoked


async def revoke_auth_session(session: AsyncSession, token: str) -> None:
    result = await session.execute(
        select(AuthSession).where(AuthSession.token_hash == hash_session_token(token))
    )
    auth_session = result.scalar_one_or_none()
    if auth_session is not None and auth_session.revoked_at is None:
        auth_session.revoked_at = utc_now()
        await session.flush()


async def get_current_user(
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    session: AsyncSession = Depends(get_db_session),
) -> AuthenticatedUser:
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    now = utc_now()
    result = await session.execute(
        select(AuthSession, User)
        .join(User, User.id == AuthSession.user_id)
        .where(
            AuthSession.token_hash == hash_session_token(session_token),
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > now,
            User.is_active.is_(True),
        )
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    auth_session, user = row
    auth_session.last_seen_at = now
    await session.commit()
    return AuthenticatedUser(
        id=user.id,
        saml_subject=user.saml_subject,
        email=user.email,
        display_name=user.display_name,
    )
