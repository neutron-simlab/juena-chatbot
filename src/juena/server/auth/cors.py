"""Allow the configured IdP's browser logout callbacks on the SLS route only."""

from urllib.parse import urlsplit

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class SamlLogoutCORSMiddleware:
    def __init__(self, app: ASGIApp, *, idp_metadata_url: str) -> None:
        self.app = app
        parsed = urlsplit(idp_metadata_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or "*" in parsed.netloc:
            raise ValueError("SAML IdP metadata must have an explicit HTTP origin")
        self.logout_app = CORSMiddleware(
            app,
            allow_origins=[f"{parsed.scheme}://{parsed.netloc}"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=True,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # IFFLogin performs SLO from its status page using browser requests.
        # Its origin must be able to read the signed reply/redirect. Giving it
        # global CORS access would also expose credentialed app API responses.
        # Authentication still comes from the SAML signature, never from Origin
        # or a session cookie. Keep normal navigation and other routes intact.
        app = self.logout_app if scope["type"] == "http" and scope["path"] == "/auth/sls" else self.app
        await app(scope, receive, send)
