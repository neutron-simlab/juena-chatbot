"""Personal service credentials. Plaintext exists only while checking or using a key."""

from __future__ import annotations

import json
from typing import Literal
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena_core.server.database.connection import get_session_factory
from juena_core.server.database.models import AuthSession, User, utc_now
from juena_core.server.identity import hash_session_token
from juena.server.database.models import UserAPICredential

Service = Literal["blablador", "tavily", "context7"]
SERVICES: tuple[Service, ...] = ("blablador", "tavily", "context7")


class CredentialError(ValueError):
    """A personal key is missing, unreadable, refused or cannot be checked.

    ``code`` and ``status`` become the error code and HTTP status the API returns.
    """

    def __init__(self, message: str, *, code: str = "credential_required", status: int = 403):
        super().__init__(message)
        self.code = code
        self.status = status


def service_endpoint(service: Service) -> str:
    """Return the address a service's key is used against.

    Args:
        service: The service.

    Returns:
        The service's base URL, without a trailing slash.
    """
    return {
        "blablador": global_config.BLABLADOR_BASE_URL,
        "tavily": "https://api.tavily.com",
        "context7": global_config.CONTEXT7_MCP_URL,
    }[service].rstrip("/")


def _encrypt(key: str, owner: UUID, service: Service, endpoint: str) -> str:
    payload = {"key": key, "owner": str(owner), "service": service, "endpoint": endpoint}
    return Fernet(global_config.JUENA_SECRET_KEY).encrypt(json.dumps(payload).encode()).decode()


def _decrypt(row: UserAPICredential, owner: UUID, service: Service) -> str:
    try:
        payload = json.loads(Fernet(global_config.JUENA_SECRET_KEY).decrypt(row.encrypted_key.encode()))
        expected = (str(owner), service, service_endpoint(service))
        if ((str(row.user_id), row.service, row.endpoint) != expected
                or (payload["owner"], payload["service"], payload["endpoint"]) != expected
                or not isinstance(payload["key"], str) or not payload["key"].strip()):
            raise ValueError
        return payload["key"]
    except (InvalidToken, ValueError, KeyError, TypeError):
        raise CredentialError(f"Replace your {service} key in My profile.", code="credential_unreadable") from None


async def credential_status(session: AsyncSession, user_id: UUID) -> dict:
    """Report, per service, whether the user has a saved key and can use it.

    Args:
        session: Database session.
        user_id: The user.

    Returns:
        For each service: ``saved``, ``available``, ``endpoint`` and
        ``validated_at``.
    """
    rows =(await session.scalars(select(UserAPICredential).where(
        UserAPICredential.user_id == user_id))).all()
    by_service = {row.service: row for row in rows}
    result = {}
    for service in SERVICES:
        row = by_service.get(service)
        available = False
        if row is not None:
            try:
                _decrypt(row, user_id, service)
                available = True
            except CredentialError:
                pass
        result[service] = {
            "saved": row is not None, "available": available,
            "endpoint": service_endpoint(service),
            "validated_at": row.validated_at.isoformat() if row else None,
        }
    return result


async def personal_key(user_id: str | None, service: Service) -> str:
    """Return the user's decrypted key for a service.

    Read fresh for every outbound call, including background recovery.

    Args:
        user_id: The signed-in user's id, as text.
        service: The service.

    Returns:
        The key.

    Raises:
        CredentialError: There is no signed-in user, the account is inactive,
            or there is no readable key.
    """
    if not user_id:
        raise CredentialError("A signed-in JüNA user is required.")
    try:
        owner = UUID(user_id)
    except (ValueError, TypeError, AttributeError):
        raise CredentialError("A signed-in JüNA user is required.") from None
    async with get_session_factory()() as session:
        active = await session.scalar(select(User.is_active).where(User.id == owner))
        row = await session.get(UserAPICredential, (owner, service))
        if not active:
            raise CredentialError("Your JüNA account is not active.")
        if row is None:
            raise CredentialError(f"Add your {service} API key in My profile.")
        return _decrypt(row, owner, service)


async def _require_live_session(session: AsyncSession, user_id: UUID, token: str) -> None:
    live = await session.scalar(select(AuthSession).where(
        AuthSession.token_hash == hash_session_token(token),
        AuthSession.user_id == user_id,
        AuthSession.revoked_at.is_(None), AuthSession.expires_at > utc_now(),
    ).with_for_update().execution_options(populate_existing=True))
    if live is None:
        raise CredentialError("Your session expired. Sign in again.", code="session_expired", status=401)


async def check_key(service: Service, key: str, endpoint: str) -> None:
    """Bounded checks without conversation content or raw upstream errors.

    Args:
        service: The service.
        key: The key to check.
        endpoint: The service address to check it against.

    Raises:
        CredentialError: The service rejected the key, or could not be reached
            or was rate limiting.
    """
    if service == "context7":
        from juena.tools.context7 import call_context7
        try:
            await call_context7(key, "resolve-library-id", {"libraryName": "python", "query": "Python documentation"})
        except Exception as error:
            if getattr(error, "status_code", None) in (401, 403):
                raise CredentialError("Context7 rejected this key.", code="credential_invalid", status=400) from None
            raise CredentialError("Context7 is unavailable or rate limited. Try again shortly.",
                                  code="credential_check_unavailable", status=503) from None
        return
    path = "/models" if service == "blablador" else "/usage"
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(endpoint + path, headers={"Authorization": f"Bearer {key}"})
        if response.status_code in (401, 403):
            raise CredentialError(f"{service.capitalize()} rejected this key.", code="credential_invalid", status=400)
        response.raise_for_status()
        body = response.json()
        expected = "data" if service == "blablador" else "key"
        if not isinstance(body, dict) or expected not in body:
            raise ValueError
    except CredentialError:
        raise
    except (httpx.HTTPError, ValueError):
        raise CredentialError(f"{service.capitalize()} is unavailable or rate limited. Try again shortly.",
                              code="credential_check_unavailable", status=503) from None


async def change_key(session: AsyncSession, *, user_id: UUID, session_token: str,
                     service: Service, endpoint: str, key: str | None) -> dict:
    """Serialize changes per owner; a failed check never replaces the saved key.

    Args:
        session: Database session.
        user_id: The key's owner.
        session_token: The session cookie; the change is refused if that
            session ended meanwhile.
        service: The service.
        endpoint: The service address the profile page showed; it must match
            the current one.
        key: The new key; None deletes the saved one.

    Returns:
        The user's key status for every service.

    Raises:
        CredentialError: The address changed, the account is inactive, the key
            was refused or could not be checked, or the session ended.
    """
    if endpoint != service_endpoint(service):
        raise CredentialError("The service address changed. Refresh My profile and try again.",
                              code="credential_endpoint_changed", status=409)
    active = await session.scalar(select(User.is_active).where(User.id == user_id).with_for_update())
    if not active:
        raise CredentialError("Your JüNA account is not active.")
    if key is not None:
        await check_key(service, key, endpoint)
    # Logout may have happened while the upstream check was running.
    await _require_live_session(session, user_id, session_token)
    row = await session.get(UserAPICredential, (user_id, service))
    if key is None:
        if row is not None:
            await session.delete(row)
    else:
        if row is None:
            row = UserAPICredential(user_id=user_id, service=service)
            session.add(row)
        row.encrypted_key = _encrypt(key, user_id, service, endpoint)
        row.endpoint = endpoint
        row.validated_at = utc_now()
    await session.commit()
    return await credential_status(session, user_id)
