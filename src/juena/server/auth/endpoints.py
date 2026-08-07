"""SAML login, callback, session, and logout endpoints."""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse, Response
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena.core.log import get_logger
from juena.server.auth import (
    SESSION_COOKIE_NAME,
    AuthenticatedUser,
    create_auth_session,
    get_current_user,
    revoke_auth_session,
    revoke_sessions_for_subject,
    upsert_saml_user,
)
from juena.server.database import get_db_session
from juena.server.database.models import SamlLoginRequest, utc_now
from juena.server.auth.saml import (
    PERSISTENT_NAME_ID_FORMAT,
    SamlAuthenticationError,
    SamlIdentity,
    get_saml_service,
)

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = get_logger(__name__)


def _safe_return_url(candidate: str | None) -> str:
    base = global_config.UI_PUBLIC_URL.rstrip("/")
    if not candidate:
        return base
    expected = urlparse(base)
    parsed = urlparse(candidate)
    if parsed.scheme != expected.scheme or parsed.netloc != expected.netloc:
        raise HTTPException(status_code=400, detail="Unsafe authentication return URL")
    return candidate


DEV_BYPASS_IDP_ENTITY_ID = "urn:juena:dev-bypass"
DEV_BYPASS_SUBJECT = "dev-local-user"


def _require_saml_service():
    """Return the SAML service, or fail with a 503 that explains why it is absent.

    ``AUTH_DEV_BYPASS`` skips ``saml_lifespan`` entirely, so ``get_saml_service``
    raises ``RuntimeError`` and every SAML route answered 500 -- an opaque
    "internal server error" for a deliberate configuration choice. That is
    especially unhelpful on ``/auth/metadata``, which is the document we owe the
    IdP administrator: the natural first thing to try returns a server error
    rather than saying the bypass is on.
    """

    if global_config.AUTH_DEV_BYPASS:
        raise HTTPException(
            status_code=503,
            detail="SAML is disabled because AUTH_DEV_BYPASS is on. Unset it to configure the SP.",
        )
    return get_saml_service()


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=global_config.SESSION_TTL_HOURS * 60 * 60,
        httponly=True,
        secure=global_config.SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.get("/login")
async def login(
    return_to: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    if global_config.AUTH_DEV_BYPASS:
        raise HTTPException(
            status_code=503,
            detail="SAML is disabled because AUTH_DEV_BYPASS is on. Use /auth/dev-login.",
        )
    relay_state = _safe_return_url(return_to)
    request_id, redirect_url = get_saml_service().prepare_login(relay_state)
    session.add(SamlLoginRequest(request_id=request_id, relay_state=relay_state))
    await session.commit()
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/dev-login")
async def dev_login(
    return_to: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """Issue a session for a fixed local user, without contacting any IdP.

    Only reachable when `AUTH_DEV_BYPASS` is set, which the configuration
    refuses to accept outside development. The session it creates is a real
    `auth_sessions` row for a real `users` row, so thread ownership, memory
    namespacing and every other authenticated path behave exactly as they do
    in production - only the identity proof is skipped.
    """

    if not global_config.AUTH_DEV_BYPASS:
        raise HTTPException(status_code=404, detail="Not found")

    logger.warning("Issuing a development session without identity verification")
    user = await upsert_saml_user(
        session,
        SamlIdentity(
            subject=DEV_BYPASS_SUBJECT,
            subject_format=PERSISTENT_NAME_ID_FORMAT,
            issuer=DEV_BYPASS_IDP_ENTITY_ID,
            in_response_to="dev-bypass",
            email="dev@localhost",
            display_name="Local Developer",
        ),
    )
    token = await create_auth_session(session, user)
    await session.commit()

    response = RedirectResponse(
        _safe_return_url(return_to),
        status_code=status.HTTP_303_SEE_OTHER,
    )
    _set_session_cookie(response, token)
    return response


@router.post("/acs")
async def assertion_consumer_service(
    saml_response: str = Form(alias="SAMLResponse"),
    relay_state: str = Form(alias="RelayState"),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """Receive the SAML login result from IFFLogin.

    IFFLogin posts the browser back to this endpoint after the user signs in.
    The SAMLResponse proves who the user is; RelayState is the original UI URL
    we saved during /auth/login, so we know where to send the browser next.
    """
    cutoff = utc_now() - timedelta(minutes=10)
    result = await session.execute(
        select(SamlLoginRequest).where(
            SamlLoginRequest.consumed_at.is_(None),
            SamlLoginRequest.created_at >= cutoff,
        )
    )
    requests = {record.request_id: record for record in result.scalars()}
    try:
        identity = get_saml_service().parse_response(
            saml_response,
            {request_id: record.relay_state for request_id, record in requests.items()},
        )
    except SamlAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    # Re-read and lock the exact request before consuming it. This closes the
    # race where two valid copies of the same assertion arrive concurrently.
    locked_result = await session.execute(
        select(SamlLoginRequest)
        .where(
            SamlLoginRequest.request_id == identity.in_response_to,
            SamlLoginRequest.consumed_at.is_(None),
            SamlLoginRequest.created_at >= cutoff,
        )
        .with_for_update()
    )
    login_request = locked_result.scalar_one_or_none()
    if login_request is None or login_request.relay_state != relay_state:
        raise HTTPException(status_code=401, detail="Unknown, expired, or replayed SAML request")

    # This is the "back to the app" location after SAML has succeeded. It is
    # not hardcoded here; it comes from the return_to URL passed into /auth/login
    # and echoed back by IFFLogin as RelayState.
    return_url = _safe_return_url(relay_state)
    login_request.consumed_at = utc_now()

    user = await upsert_saml_user(session, identity)
    token = await create_auth_session(session, user)
    await session.commit()

    response = RedirectResponse(return_url, status_code=status.HTTP_303_SEE_OTHER)
    _set_session_cookie(response, token)
    return response


@router.get("/me")
async def current_user(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str | None]:
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
    }


@router.post("/logout")
async def logout(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Application logout for the signed-in UI session.

    This is not the SAML SingleLogoutService; the IdP talks to ``/auth/sls``.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await revoke_auth_session(session, token)
        await session.commit()
    redirect_url: str | None = None
    try:
        redirect_url = get_saml_service().prepare_logout(user.saml_subject)
    except Exception:
        # Local session revocation is authoritative and must succeed even when
        # the IdP has no SLO endpoint or rejects the logout request.
        logger.warning("FZJ single logout failed after local revocation", exc_info=True)
    response: Response
    if redirect_url:
        response = RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    else:
        # A browser-submitted logout should return to the UI even when the IdP
        # does not expose SLO (including the local development bypass).
        response = RedirectResponse(
            global_config.UI_PUBLIC_URL.rstrip("/"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@router.get("/sls")
@router.post("/sls")
async def single_logout_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """SAML SingleLogoutService.

    Public and unauthenticated by necessity: the IdP posts here without our
    cookie, both to answer our own LogoutRequest and to initiate logout of its
    own accord. Everything is authorized by SAML signature validation.
    """

    form = await request.form() if request.method == "POST" else {}
    binding = BINDING_HTTP_POST if request.method == "POST" else BINDING_HTTP_REDIRECT
    params = form or request.query_params
    relay_state = params.get("RelayState")
    saml_service = _require_saml_service()

    if saml_response := params.get("SAMLResponse"):
        try:
            saml_service.parse_logout_response(saml_response, binding)
        except SamlAuthenticationError:
            logger.warning("Discarding invalid SAML logout response", exc_info=True)
        return _post_logout_redirect(relay_state)

    saml_request = params.get("SAMLRequest")
    if not saml_request:
        raise HTTPException(status_code=400, detail="Missing SAMLRequest or SAMLResponse")

    try:
        subject, redirect_url = saml_service.parse_logout_request(
            saml_request,
            binding,
            relay_state=relay_state,
        )
    except SamlAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if subject:
        await revoke_sessions_for_subject(session, subject)
        await session.commit()

    response: Response = (
        RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
        if redirect_url
        else _post_logout_redirect(relay_state)
    )
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


def _post_logout_redirect(relay_state: str | None) -> RedirectResponse:
    """Send the browser back to the UI after a completed logout."""

    try:
        target = _safe_return_url(relay_state)
    except HTTPException:
        target = global_config.UI_PUBLIC_URL.rstrip("/")
    response = RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@router.get("/metadata")
async def service_provider_metadata() -> Response:
    return Response(
        content=_require_saml_service().metadata_xml(),
        media_type="application/samlmetadata+xml",
    )
