# IFFLogin (SAML2) integration — status and open items

Working notes for connecting JueNA to the Forschungszentrum Jülich IFFLogin
identity provider. Summarises the agreement with Daniel (IFFLogin admin), what
was verified against the live IdP metadata, and what still blocks going live.

- **Conversation:** 19 November 2025
- **Metadata verified:** 5 August 2026, against <https://ifflogin.fz-juelich.de/saml/idp.xml>

---

## 1. What the conversation settled

| Question | Outcome |
| --- | --- |
| OAuth2 or SAML2? | **SAML2.** Daniel: "SAML2 has many advantages (e.g. Single Log-Out) and would be preferred." |
| Where are the IdP's capabilities documented? | The IdP metadata URL above — bindings, NameID formats, certificates. |
| What does JueNA have to provide? | **SP metadata for our service**, followed by "some negotiation about the used bindings internally." |

Two things follow from this that shape the implementation:

1. **Single Log-Out is the stated reason SAML2 was chosen**, so SLO has to
   actually work — it is not optional polish.
2. **We owe Daniel an SP metadata document.** That is the concrete next
   deliverable, and it cannot be produced until the deployment hostname exists
   (see §5).

---

## 2. Historical IdP metadata snapshot

Read directly from the live metadata on 5 August 2026.

| Property | Value |
| --- | --- |
| Entity ID | `https://ifflogin.fz-juelich.de/saml/idp.xml` |
| Signed AuthnRequests | `WantAuthnRequestsSigned="true"` — **required** |
| NameID formats | `transient`, **`persistent`**, `emailAddress` (SAML 1.1) |
| SSO — HTTP-POST | `https://ifflogin.fz-juelich.de/saml/sso/post` |
| SSO — HTTP-Redirect | `https://ifflogin.fz-juelich.de/saml/sso/redirect` |
| SLO — HTTP-POST | `https://ifflogin.fz-juelich.de/saml/slo/post` |
| SLO — HTTP-Redirect | `https://ifflogin.fz-juelich.de/saml/slo/redirect` |
| Key descriptor | One, `use="signing"` |
| Metadata signature | **Signed** |
| `validUntil` | `2026-08-12T15:38:20Z` — see the warning below |
| Declared attributes | **None** — the metadata lists no `<Attribute>` elements |
| Contacts published | `Organization`, plus `technical` and `support` `ContactPerson` |
| Algorithms advertised | Digest and signing methods up to SHA-512, including `rsa-sha256` |

Both Single Log-Out bindings were advertised in that snapshot. Current support
and end-to-end behavior must be confirmed for this development SP.

> **The values above are a snapshot taken on 5 August 2026 and the metadata
> expires on 12 August 2026.** Two consequences. First, re-read the live
> document before relying on any row in this table — a refreshed `idp.xml`
> carries a new `validUntil` and may carry new certificates. Second, the roughly
> one-week validity window is not a quirk of this snapshot but the IdP's normal
> cadence, which requires periodic refresh: a deployment
> that fetches metadata only at startup stops recognising the IdP mid-run, and
> every login fails with no deploy to explain it.

---

## 3. Where JueNA already lines up

| Aspect | IdP expects | JueNA sends | |
| --- | --- | --- | --- |
| AuthnRequest signing | Required | `authn_requests_signed: True` | ✅ |
| NameID format | persistent supported | persistent requested, and enforced on response | ✅ |
| SSO bindings | POST + Redirect | Redirect out, POST back | ✅ |
| SLO bindings | POST + Redirect | Both, on `/auth/sls` | ✅ |
| IdP entity ID | as above | matches `SAML_IDP_METADATA_URL` | ✅ |

Endpoint layout ([auth_endpoints.py](src/juena/server/auth/endpoints.py)):

| Endpoint | Purpose |
| --- | --- |
| `GET /auth/login` | Start SP-initiated login (signed AuthnRequest). |
| `POST /auth/acs` | AssertionConsumerService; validates and sets the session cookie. |
| `GET /auth/metadata` | **The SP metadata to send Daniel.** |
| `POST /auth/logout` | Application logout for the signed-in UI session. |
| `GET\|POST /auth/sls` | SAML SingleLogoutService. |

> `/auth/logout` and `/auth/sls` are deliberately separate. The IdP reaches
> `/auth/sls` **without our cookie** — both to answer our LogoutRequest and to
> start logout itself — and is authorised by signature validation instead.
> `/auth/logout` requires an authenticated session. Pointing `SAML_SP_SLS_URL`
> at `/auth/logout` makes IdP-initiated logout fail with 401.

---

## 4. Implemented readiness work and remaining confirmations

The application explicitly configures RSA-SHA256 and SHA-256 under PySAML2's
`service.sp` configuration, where the client reads them. Tests inspect the
actual Redirect `SigAlg`, not just the configuration dictionary.

Production requires `SAML_IDP_METADATA_CERT_FILE`. The remote metadata signature
is verified against this certificate, and unsigned metadata is explicitly
rejected (PySAML2 otherwise accepts unsigned metadata even when `cert` is set).
TLS verification remains enabled. Obtain the trust certificate independently
from IFFLogin; do not trust a certificate merely because the downloaded XML
contains it.

`SAML_METADATA_REFRESH_SECONDS` defaults to six hours. Refresh keeps the existing
client and its login/logout state, replaces verified metadata in-process, and
retries failed fetches. Expired metadata blocks SAML operations. Startup requires
fresh metadata and has no stale disk fallback. Coordinate metadata trust-anchor
changes with the administrator; assertion-key changes arrive through verified
metadata. Confirm the required refresh and rollover policy with IFFLogin.

Set `SAML_ORGANIZATION_NAME`, `SAML_ORGANIZATION_URL`, `SAML_CONTACT_NAME` and
`SAML_CONTACT_EMAIL` to include the SP organization and technical contact.
Email/display attributes are optional profile fields; confirm their exact wire
names. Persistent NameID is required for identity.

Real IFFLogin browser login, authenticated session, logout and repeat login are
still required before declaring SAML ready. The historical snapshot above was
not revalidated during this deployment work.

---

## 5. Deployment prerequisites

### 5.1 The SP base URL

`https://juena.fz-juelich.de` — the **public HTTPS origin where FastAPI is
reachable from the user's browser**, not from inside Docker. SAML redirects the
browser, so the address must resolve on the user's machine.

You administer the Proxmox VM. Confirm hostname allocation and TLS issuance only
where institutionally controlled; do not assume the IFFLogin administrator owns
those services. Replace this example hostname with the agreed development host:

```bash
SAML_IDP_METADATA_CERT_FILE=/data/saml/idp-metadata.crt
SAML_SP_ENTITY_ID=https://juena.fz-juelich.de/auth/metadata
SAML_SP_ACS_URL=https://juena.fz-juelich.de/auth/acs
SAML_SP_SLS_URL=https://juena.fz-juelich.de/auth/sls
JUENA_API_INTERNAL_URL=http://localhost:8080   # see note below
JUENA_PUBLIC_URL=https://juena.fz-juelich.de
UI_PUBLIC_URL=https://juena.fz-juelich.de
SESSION_COOKIE_SECURE=true
ENVIRONMENT=production
```

`JUENA_API_INTERNAL_URL` is private server-to-server configuration and is never
placed in browser links. `JUENA_PUBLIC_URL` is the public origin used for login
links. `UI_PUBLIC_URL` is a one-release compatibility setting for the post-login
destination; in production it defaults to `JUENA_PUBLIC_URL`. See §5.2.

> `JUENA_API_INTERNAL_URL=http://localhost:8080` is correct **only while FastAPI
> and Streamlit share one container**, which is the current layout. Splitting
> them into separate `api` and `ui` services would make `localhost` inside the
> UI container the UI itself, and the value would become `http://api:8080`.
> Change it in the same release as any such split; nothing else in this file is
> affected, because every other URL here is browser-facing.

### 5.2 Both services must share one hostname

`/auth/acs` sets `juena_session` with `path="/"` and **no `domain=`**
([`_set_session_cookie`](src/juena/server/auth/endpoints.py#L53-L62)), so the
cookie is host-only. Streamlit then reads it through `st.context.cookies`.

Splitting the two across subdomains — FastAPI on `api.…`, Streamlit on
`chat.…` — means the cookie is never sent to Streamlit, and login fails
silently: the user round-trips through the IdP and lands back logged out. It
works locally only because `localhost:8080` and `localhost:9501` are the same
host; cookies ignore the port.

So both run behind `juena.fz-juelich.de`, split by path in the reverse proxy:

| Path | Backend |
| --- | --- |
| `/auth/*`, `/chats/*`, `/threads/*`, `/artifacts/*`, `/research/*`, `/health`, and optional-agent `stream`, `stream_with_files`, `resume` paths | FastAPI `:8080` |
| everything else, including `/_stcore/*` | Streamlit `:9501` |

That is why `JUENA_PUBLIC_URL`, `UI_PUBLIC_URL`, and the SP base URL are the
same production origin.

### 5.3 Two different certificates — do not conflate them

This is the easiest thing to get confused about, because both are "the
certificate" in casual conversation.

| | TLS / HTTPS certificate | SAML SP signing certificate |
| --- | --- | --- |
| **Purpose** | Browsers trust `https://juena.fz-juelich.de` | Signs our AuthnRequests and LogoutRequests |
| **Seen by** | Every visitor's browser | The IdP only |
| **Published in** | The TLS handshake | Our SP metadata (`/auth/metadata`) |
| **Who provides it** | Daniel / FZJ IT, with the hostname | **Open question** — normally the SP generates its own |
| **Rotation** | Renewed routinely, invisible to the IdP | Changing it means **re-registering with the IdP** |

Daniel taking care of "the URL and its certificate" almost certainly means the
**TLS** certificate. The SAML signing keypair is a separate artefact, and by
convention the service provider generates it and publishes only the public half
in its metadata. Worth confirming explicitly rather than assuming — see §6.

Either way, the current situation is development-only: the Docker entrypoint
generates a throwaway **30-day self-signed** keypair under `/data/saml`.
Production needs a long-lived certificate that stays stable, mounted via
`SAML_SP_CERT_FILE` and `SAML_SP_KEY_FILE`, and backed up — losing the private
key means re-registering the SP.

### 5.4 Internal and browser-facing URLs

The hardcoded backend URL blocker is resolved. Streamlit uses
`JUENA_API_INTERNAL_URL` for pooled server-to-server FastAPI requests and builds
browser login links from `JUENA_PUBLIC_URL`. The internal value is not stored in
chat session controls or rendered in the UI.

Development keeps sensible split-port defaults (`8080` for FastAPI, `9501` for
Streamlit). Production has no localhost public default: startup rejects missing
or non-HTTPS public/SAML URLs and rejects an insecure session cookie.

The base Compose file does not publish PostgreSQL;
[docker-compose.dev.yml](docker-compose.dev.yml) does. That file is **not** an
auto-merged `docker-compose.override.yml` — it must be passed explicitly:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d   # or: juena up
```

Omitting the second `-f` silently starts the production configuration against
your development `.env`, which is exactly how a container ends up dialling
`localhost:5432` for a Postgres that lives in another container. `juena up`
always passes both.

---

## 6. Running locally while registration is pending

The IdP will not accept an unregistered SP, so no real login is possible yet.
`AUTH_DEV_BYPASS` provides a local path in the meantime.

```bash
# .env
ENVIRONMENT=development
AUTH_DEV_BYPASS=true
DATABASE_URL=postgresql://juena:juena-development-only@localhost:5432/juena
```

```bash
juena postgres-up
uv run main.py          # or use docker-compose.yml + docker-compose.dev.yml
```

The sign-in button then points at `/auth/dev-login`, which issues a session for
a fixed `Local Developer` account.

**What it does and does not change:**

| | Behaviour |
| --- | --- |
| Identity proof | **Skipped** — no assertion is verified |
| `users` / `auth_sessions` rows | Real, created through the normal code path |
| Thread ownership, memory namespacing | Unchanged — production logic throughout |
| Unauthenticated requests | Still `401`; the bypass grants a session, it does not disable auth |
| `/auth/login`, `/auth/metadata`, `/auth/sls` | Return `503` naming the bypass; `saml_lifespan` is skipped entirely |
| SP certificates, `xmlsec1` | **Not required** — so native macOS runs work |

> Note for producing the SP metadata: `/auth/metadata` **cannot** emit a document
> while the bypass is on, because no SP is configured to describe. Unset
> `AUTH_DEV_BYPASS`, provide the SP keypair, and restart before running
> `juena sp-metadata`. Until then the endpoint answers `503` with that reason —
> it previously answered an opaque `500`, which looked like a crash.

The dev user is keyed on a fixed persistent subject, so its UUID, chats and
memories survive restarts.

**Safety:** `ENVIRONMENT=production` together with `AUTH_DEV_BYPASS=true` raises
at import and the process will not start. This is deliberate — a flag that
issues sessions without verifying identity must fail loudly rather than be
silently inherited into a production deployment.

Delete `AUTH_DEV_BYPASS` from the environment once the SP is registered.

---

## 7. External confirmations

Ask IFFLogin about a separate development SP for an intranet/VPN-only hostname,
metadata submission versus direct URL fetching, persistent NameID, optional
email/display-name attribute names, RSA-SHA256/SHA-256, signed responses and
assertions, Redirect-out/POST-back SSO, and front-channel SLO behavior.
Obtain the metadata verification certificate, refresh/rollover procedure, SP
keypair/lifetime policy, and pilot-user eligibility agreement.

Confirm DNS, TLS termination/renewal, upstream institute/VPN rules and shared
Blablador credentials only with whichever institutional team controls them.
You own Docker, Podman, storage, systemd and the VM firewall on Proxmox.
The complete questions and answer fields are in the local ignored
`DOCS/DEV-SERVER-CHECKLIST.md`.

---

## 8. Acceptance status

- [x] RSA-SHA256 signing and SHA-256 digests, checked on an actual Redirect.
- [x] Verified signed metadata and production trust-certificate requirement.
- [x] Periodic refresh preserving the client state, with expiry rejection.
- [x] Configurable organization and technical contact in SP metadata.
- [x] One HTTPS gateway for internal API/UI routes and public browser URLs.
- [ ] Agree hostname, trust material, registration and eligibility with IFFLogin.
- [ ] Real browser login, authenticated session, logout and repeat login.
- [ ] Verify IdP-initiated SLO, certificate rollover and metadata refresh on the VM.

Logout verifies Redirect signatures or POST XML signatures before revoking
sessions, correlates responses with pending requests, and can return either a
redirect or a POST form. Actual IFFLogin interoperability is still pending.
The local PySAML2 login/logout cache is in memory, so test logout after an app
restart as part of the IdP agreement; durable local app sessions alone do not
preserve the IdP's session index across process restarts.

---

## 9. References

- IdP metadata: <https://ifflogin.fz-juelich.de/saml/idp.xml>
- PySAML2 configuration: <https://pysaml2.readthedocs.io/en/latest/howto/config.html>
- SP implementation: [saml_service.py](src/juena/server/auth/saml.py),
  [auth_endpoints.py](src/juena/server/auth/endpoints.py)
- Settings: [config.py](src/juena/core/config.py), [env.example](env.example)
