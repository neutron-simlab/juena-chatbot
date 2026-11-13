"""SAML login, callback, session, and logout endpoints."""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlparse

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from juena.core.config import global_config
from juena_core.log import get_logger
from juena.server.auth import (
    SESSION_COOKIE_NAME,
    Principal,
    create_auth_session,
    session_principal,
    revoke_auth_session,
    revoke_sessions_for_subject,
    upsert_saml_user,
)
from juena_core.server.database.connection import get_db_session
from juena_core.server.database.models import utc_now
from juena.server.database.models import SamlLoginRequest
from juena.server.auth.saml import (
    PERSISTENT_NAME_ID_FORMAT,
    SamlAuthenticationError,
    SamlIdentity,
    SamlLogoutMessage,
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
    would raise ``RuntimeError`` and the route would answer an opaque 500 for a
    deliberate configuration choice. This matters most on ``/auth/metadata``,
    the document the IdP administrator asks for.
    """

    if global_config.AUTH_DEV_BYPASS:
        raise HTTPException(
            status_code=503,
            detail="SAML is disabled because AUTH_DEV_BYPASS is on. Unset it to configure the SP.",
        )
    return get_saml_service()


#: Ties a sign-in to the browser that started it; holds the SAML request id.
#: The `__Host-` prefix makes browsers refuse this cookie from any other host,
#: so a neighbouring site under the same parent domain cannot plant one.
LOGIN_COOKIE_NAME = "__Host-juena_login"
#: How long IFFLogin's answer to a sign-in request is accepted.
LOGIN_REQUEST_TTL_MINUTES = 10
# Shared by setting and clearing: a browser ignores a `__Host-` cookie that is
# not Secure with path "/", and that includes the instruction to delete it.
# SameSite is "none" because IFFLogin posts its answer from its own site, and
# "lax" would make the browser withhold the cookie from exactly that request.
_LOGIN_COOKIE_ATTRIBUTES = {"path": "/", "secure": True, "httponly": True, "samesite": "none"}


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
    """Start a SAML sign-in: record the request and send the browser to IFFLogin.

    Args:
        return_to: UI address to come back to; it must be on the UI's origin.
        session: Database session.

    Returns:
        A redirect to IFFLogin that also sets the login cookie.

    Raises:
        HTTPException: 503 when SAML is off or the IdP metadata has expired;
            400 for an unsafe return address.
    """
    if global_config.AUTH_DEV_BYPASS:
        raise HTTPException(
            status_code=503,
            detail="SAML is disabled because AUTH_DEV_BYPASS is on. Use /auth/dev-login.",
        )
    relay_state = _safe_return_url(return_to)
    request_id, redirect_url = get_saml_service().prepare_login(relay_state)
    # This route needs no sign-in and writes a row each time it is called, so
    # it also clears the rows that are too old for `/auth/acs` to accept.
    cutoff = utc_now() - timedelta(minutes=LOGIN_REQUEST_TTL_MINUTES)
    await session.execute(delete(SamlLoginRequest).where(SamlLoginRequest.created_at < cutoff))
    session.add(SamlLoginRequest(request_id=request_id, relay_state=relay_state))
    await session.commit()
    response = RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        LOGIN_COOKIE_NAME,
        request_id,
        max_age=LOGIN_REQUEST_TTL_MINUTES * 60,
        **_LOGIN_COOKIE_ATTRIBUTES,
    )
    return response


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

    Args:
        return_to: UI address to come back to; it must be on the UI's origin.
        session: Database session.

    Returns:
        A redirect to the UI that sets the session cookie.

    Raises:
        HTTPException: 404 when the bypass is off; 400 for an unsafe return
            address.
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
    login_cookie: str | None = Cookie(default=None, alias=LOGIN_COOKIE_NAME),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """Receive the SAML login result from IFFLogin.

    IFFLogin posts the browser back to this endpoint after the user signs in.
    The SAMLResponse proves who the user is; RelayState is the original UI URL
    we saved during /auth/login, so we know where to send the browser next.

    Args:
        saml_response: The signed sign-in response.
        relay_state: The UI address saved when the sign-in started.
        login_cookie: The login cookie set by ``/auth/login``.
        session: Database session.

    Returns:
        A redirect to the UI that sets the session cookie.

    Raises:
        HTTPException: 401 when the response is invalid, was started in another
            browser, or answers an unknown, expired or used request.
    """
    cutoff = utc_now() - timedelta(minutes=LOGIN_REQUEST_TTL_MINUTES)
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

    # The answer is valid, but it travels through the browser, so the person
    # who signed in could have copied it and made someone else's browser
    # deliver it. Only the browser that started this sign-in holds its cookie.
    if login_cookie != identity.in_response_to:
        raise HTTPException(
            status_code=401,
            detail="This sign-in was not started in this browser. Open JüNA and sign in again.",
        )

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
    response.delete_cookie(LOGIN_COOKIE_NAME, **_LOGIN_COOKIE_ATTRIBUTES)
    return response


@router.get("/me")
async def current_user(response: Response, user: Principal = Depends(session_principal),
                       session: AsyncSession = Depends(get_db_session)) -> dict:
    """Return the signed-in user and the status of their personal keys.

    Args:
        response: Used to forbid caching.
        user: The signed-in user.
        session: Database session.

    Returns:
        The user's id, email, display name, key status per service, and
        whether they can chat.
    """
    from juena.server.credentials import credential_status

    response.headers["Cache-Control"] = "no-store"
    credentials = await credential_status(session, user.id)
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "credentials": credentials,
        "chat_ready": credentials["blablador"]["available"],
    }


@router.post("/logout")
async def logout(
    request: Request,
    user: Principal = Depends(session_principal),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Application logout for the signed-in UI session.

    This is not the SAML SingleLogoutService; the IdP talks to ``/auth/sls``.

    Args:
        request: The browser request, carrying the session cookie.
        user: The signed-in user.
        session: Database session.

    Returns:
        The IdP's logout message when single logout is possible, otherwise a
        redirect to the UI. Either way the session cookie is cleared.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await revoke_auth_session(session, token)
        await session.commit()
    message: SamlLogoutMessage | None = None
    try:
        message = get_saml_service().prepare_logout(user.subject)
    except Exception:
        # Local session revocation is authoritative and must succeed even when
        # the IdP has no SLO endpoint or rejects the logout request.
        logger.warning("FZJ single logout failed after local revocation", exc_info=True)
    response: Response
    if message:
        response = _logout_message_response(message)
    else:
        # A browser-submitted logout should return to the UI even when the IdP
        # does not expose SLO (including the local development bypass).
        response = RedirectResponse(
            global_config.UI_PUBLIC_URL.rstrip("/"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


def _logout_message_response(message: SamlLogoutMessage) -> Response:
    if message.redirect_url:
        return RedirectResponse(message.redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    return HTMLResponse(message.html or "", headers={"Cache-Control": "no-store"})


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

    Args:
        request: The IdP's message, as query parameters or a posted form.
        session: Database session.

    Returns:
        Our LogoutResponse for the IdP when one is due, otherwise a redirect
        to the UI.

    Raises:
        HTTPException: 400 when neither message is present, 401 when a message
            fails validation, 503 when SAML is off.
    """

    form = await request.form() if request.method == "POST" else {}
    binding = BINDING_HTTP_POST if request.method == "POST" else BINDING_HTTP_REDIRECT
    params = form or request.query_params
    relay_state = params.get("RelayState")
    saml_service = _require_saml_service()

    if saml_response := params.get("SAMLResponse"):
        try:
            saml_service.parse_logout_response(
                saml_response, binding, relay_state=relay_state,
                sigalg=params.get("SigAlg"), signature=params.get("Signature"),
            )
        except SamlAuthenticationError as exc:
            logger.warning("SAML logout response rejected (%s)", type(exc.__cause__ or exc).__name__)
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return _post_logout_redirect(relay_state)

    saml_request = params.get("SAMLRequest")
    if not saml_request:
        raise HTTPException(status_code=400, detail="Missing SAMLRequest or SAMLResponse")

    try:
        subject, message = saml_service.parse_logout_request(
            saml_request,
            binding,
            relay_state=relay_state,
            sigalg=params.get("SigAlg"),
            signature=params.get("Signature"),
        )
    except SamlAuthenticationError as exc:
        logger.warning("SAML logout request rejected (%s)", type(exc.__cause__ or exc).__name__)
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if subject:
        await revoke_sessions_for_subject(session, subject)
        await session.commit()

    response: Response = (
        _logout_message_response(message)
        if message
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
    """Serve this SP's signed SAML metadata, the document the IdP registers.

    Returns:
        The metadata XML.

    Raises:
        HTTPException: 503 when SAML is off or the IdP metadata has expired.
    """
    return Response(
        content=_require_saml_service().metadata_xml(),
        media_type="text/xml",
        headers={
            "Content-Disposition": 'inline; filename="metadata.xml"',
            "Cache-Control": "no-cache",
        },
    )
