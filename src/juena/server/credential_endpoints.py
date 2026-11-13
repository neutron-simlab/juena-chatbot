"""Owner-only credential management; SAML endpoints keep their existing policy."""

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena.server.credentials import CredentialError, Service, change_key, personal_key
from juena_core.server.database.connection import get_db_session
from juena_core.server.identity import SESSION_COOKIE_NAME, Principal, session_principal

router = APIRouter(prefix="/auth/me/credentials", tags=["profile"])


class CredentialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: SecretStr
    endpoint: str


def _origin(request: Request) -> None:
    allowed = {f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
               for url in (global_config.UI_PUBLIC_URL, global_config.JUENA_PUBLIC_URL)}
    if request.headers.get("origin") not in allowed:
        raise HTTPException(403, {"code": "origin_not_allowed", "message": "Browser origin is not allowed."})


def _refusal(error: CredentialError) -> HTTPException:
    return HTTPException(error.status, {"code": error.code, "message": str(error)})


async def require_chat_credentials(user: Principal, provider: str | None) -> None:
    """Refuse a chat run unless the user has a usable Blablador key.

    Args:
        user: The signed-in user.
        provider: The provider the run asks for, if any.

    Raises:
        HTTPException: 400 for a provider other than Blablador, or the
            credential error's status when the key is missing or unreadable.
    """
    if provider is not None and provider != "blablador":
        raise HTTPException(400, "JüNA supports Blablador personal keys only.")
    try:
        await personal_key(str(user.id), "blablador")
    except CredentialError as error:
        raise _refusal(error) from None


@router.put("/{service}")
async def save_key(service: Service, request: Request, response: Response,
                   user: Principal = Depends(session_principal),
                   session: AsyncSession = Depends(get_db_session)) -> dict:
    """Check a personal key with its service, then save it encrypted.

    Args:
        service: Which service the key is for.
        request: The browser request; it must come from the UI's or the API's origin.
        response: Used to forbid caching.
        user: The signed-in user.
        session: Database session.

    Returns:
        The user's key status for every service.

    Raises:
        HTTPException: 403 for another origin, 400 for a malformed key, or the
            credential error's status when the check or the save is refused.
    """
    _origin(request)
    response.headers["Cache-Control"] = "no-store"
    # Parse here so validation errors cannot echo a submitted secret in a 422.
    try:
        payload = CredentialInput.model_validate(await request.json())
        key = payload.key.get_secret_value().strip()
        if not key or len(key) > 4096 or any(c.isspace() for c in key):
            raise ValueError
    except (ValidationError, ValueError):
        raise HTTPException(400, "Enter an API key and the displayed service address.") from None
    try:
        return await change_key(session, user_id=user.id,
                                session_token=request.cookies[SESSION_COOKIE_NAME], service=service,
                                endpoint=payload.endpoint, key=key)
    except CredentialError as error:
        raise _refusal(error) from None


@router.delete("/{service}")
async def remove_key(service: Service, request: Request, response: Response, endpoint: str = Query(...),
                     user: Principal = Depends(session_principal),
                     session: AsyncSession = Depends(get_db_session)) -> dict:
    """Delete the user's saved key for a service.

    Args:
        service: Which service the key is for.
        request: The browser request; it must come from the UI's or the API's origin.
        response: Used to forbid caching.
        endpoint: The service address the profile page showed.
        user: The signed-in user.
        session: Database session.

    Returns:
        The user's key status for every service.

    Raises:
        HTTPException: 403 for another origin, or the credential error's
            status when the change is refused.
    """
    _origin(request)
    response.headers["Cache-Control"] = "no-store"
    try:
        return await change_key(session, user_id=user.id,
                                session_token=request.cookies[SESSION_COOKIE_NAME], service=service,
                                endpoint=endpoint, key=None)
    except CredentialError as error:
        raise _refusal(error) from None
