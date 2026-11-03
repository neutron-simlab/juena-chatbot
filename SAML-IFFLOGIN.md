# IFFLogin (SAML2) setup and acceptance

JueNA uses the Forschungszentrum Jülich IFFLogin identity provider. This guide
records the agreed SAML configuration, the ACS fixes and the remaining browser
acceptance checks. SP registration and metadata trust setup are complete for
the development server. The user reported the login issue resolved after the
3 November 2026 fix; logout and repeat-login acceptance still need recording.

Use [deploy/SERVER.md](deploy/SERVER.md) for deployment in either gateway mode,
and [ACS troubleshooting](deploy/TROUBLESHOOTING.md#ifflogin-returns-401-at-authacs)
for the failure encountered. The SAML implementation uses configured public
URLs and has no dependency on the shared nginx project.

- **Conversation:** 19 November 2025
- **Metadata verified:** 5 August 2026, against <https://ifflogin.fz-juelich.de/saml/idp.xml>
- **Deployment confirmation reported:** 3 November 2026 — Daniel approved using
  `iff1122`; the development origin is `https://iff1122.iff.kfa-juelich.de`.
  He described nginx TLS termination with internal HTTP and hostname routing.
- **Registration prerequisite reported:** Daniel needs proper SP metadata at
  the HTTPS metadata URL before configuring the IdP.
- **SP registration confirmed, 3 November 2026:** the user reports Daniel has
  registered JueNA after receiving its SP metadata. This completes the IdP-side
  registration step; the full browser acceptance sequence remains to be recorded.
- **IdP trust confirmed, 3 November 2026:** Daniel confirmed the fingerprint
  sent by the user for IFFLogin's metadata-signing certificate. The current
  certificate matches; live XML signature verification and actual `SamlService`
  initialization passed. The app is now serving public API/UI HTTPS. Use
  `sudo ./juena prod up` for ongoing deployment; see
  [server operations](deploy/SERVER.md).
- **SP credential prepared, 3 November 2026:** user authorized the observed
  IFFLogin approach: a separate self-signed RSA-4096/SHA-256 certificate, valid
  until 3 November 2029. Public SP XML was first published through metadata-only
  nginx routing and independently verified. The live app now serves the signed
  metadata with inline XML browser headers using the same registered keypair.
- **Login callback fix, 3 November 2026:** the browser reached ACS,
  where logs reported `Signature missing for assertion`. IFFLogin's signed
  Response passed signature verification, but JueNA also required a separate
  Assertion signature. The SP now requires the Response signature and treats
  an additional Assertion signature as optional, verifying it when present.
  The signed Response covers the contained assertion. Metadata advertises
  `WantAssertionsSigned="false"`; the registered entity, endpoints and keypair
  are unchanged. Removed a call to `is_authenticated()`, which does not exist
  on PySAML2 7.5.4's validated `AuthnResponse`. Deploy with
  `sudo ./juena prod up`, then start a fresh login from the app (do not repost
  the old ACS form). The user subsequently reported the issue resolved. This
  report does not establish logout, repeat login or IdP-initiated SLO acceptance.

---

## 1. What the conversation settled

| Question | Outcome |
| --- | --- |
| OAuth2 or SAML2? | **SAML2.** Daniel: "SAML2 has many advantages (e.g. Single Log-Out) and would be preferred." |
| Where are the IdP's capabilities documented? | The IdP metadata URL above — bindings, NameID formats, certificates. |
| What does JueNA have to provide? | **SP metadata for our service**, followed by "some negotiation about the used bindings internally." |
| Existing development hostname | **Confirmed:** Daniel says `iff1122` can be used (reply reported 3 November 2026). |
| Reverse proxy | Daniel recommends nginx TLS termination, internal HTTP and routing by the requested hostname; this matches the shared-gateway configuration. |
| SP registration | **Complete as reported by Daniel via the user, 3 November 2026.** No repeat SP metadata submission/registration is currently requested. |
| Login initiation | Daniel confirms SP-initiated login only, with no JueNA listing on the IFFLogin start page. Existing `/auth/login` and `/auth/acs` implement this flow. |
| Default access | User requested JCNS staff initially. Daniel's offered internal-LDAP option covers PGI/JCNS together; JCNS-only scope and actual IdP configuration remain to be confirmed. |

Two things follow from this that shape the implementation:

1. **Single Log-Out is the stated reason SAML2 was chosen**, so SLO has to
   actually work — it is not optional polish.
2. **We owe Daniel an SP metadata document.** The hostname is now confirmed.
   He requires it at our HTTPS URL before registration. Offline SP export and
   nginx static serving allow publication before the application has verified
   IdP trust material. Registration is complete; browser login acceptance remains.

SP registration is confirmed by Daniel via the user. Production app startup
and IdP metadata trust setup are complete; the browser acceptance checklist
below still needs completing. The pinning of the IdP metadata-signing certificate is a JueNA
deployment requirement, not an additional SP registration request from Daniel.
For terminology, TLS SNI selects the certificate during the handshake; the HTTP
`Host` and nginx `server_name` select HTTP routing. See the
[nginx HTTPS/SNI documentation](https://nginx.org/en/docs/http/configuring_https_servers.html#sni).

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

IFFLogin advertises `transient`, `persistent` and `emailAddress` because an IdP
can issue different identifier formats to different applications. JueNA's SP
metadata advertises only `persistent`: its login request explicitly asks for
that format and response parsing rejects other formats. Persistent identifiers
let the app recognize the same user across logins and retain ownership of chats
and files. A transient identifier is temporary; an email address can change.
Optional email/display-name attributes remain separate from the primary NameID.
Advertising additional formats would require identity-mapping and validation
changes, rather than just adding XML tags. See the
[Shibboleth format-selection documentation](https://shibboleth.atlassian.net/wiki/spaces/IDP5/pages/3199507810).

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

The callback fix has regression coverage using signed SAML responses: a signed
Response can contain an unsigned Assertion, while unsigned Responses, tampered
XML, an untrusted key, wrong destination/audience/request and transient NameID
are rejected. A different configured public hostname is also tested. These
checks complement the real IFFLogin browser acceptance sequence; they do not
replace it. The August metadata table is historical, while the November trust
preflight fetched and verified fresh metadata.

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
([`_set_session_cookie`](src/juena/server/auth/endpoints.py)), so the
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

For a new dedicated server, configure the public app and SAML URLs for its
hostname, install the matching TLS certificate and coordinate the changed
SP metadata with IFFLogin. Select the launcher's standalone gateway as described
in [SERVER.md](deploy/SERVER.md). The signed-Response policy and PySAML2 callback
fix are the same in both proxy modes.

### 5.3 Two different certificates — do not conflate them

This is the easiest thing to get confused about, because both are "the
certificate" in casual conversation.

| | TLS / HTTPS certificate | SAML SP signing certificate |
| --- | --- | --- |
| **Purpose** | Browsers trust the HTTPS hostname | Signs SP metadata, AuthnRequests and LogoutRequests |
| **Seen by** | Every visitor's browser | The IdP only |
| **Published in** | The TLS handshake | Our SP metadata (`/auth/metadata`) |
| **Who provides it** | Approved institutional TLS issuer | Persistent self-signed SP credential generated on this VM, as authorized |
| **Rotation** | Renewed routinely, invisible to the IdP | Coordinate an SP metadata/trusted-key update with IFFLogin |

Confirm TLS issuance and renewal with the team controlling the hostname. The
SAML signing keypair is separate: the service provider retains its private key
and publishes its public certificate in metadata.

The development entrypoint can generate a throwaway **30-day self-signed**
keypair under `/data/saml`. The VM uses a separate persistent
RSA-4096/SHA-256 keypair installed under `/etc/juena/saml/`, with the private
directory mode 0700 and key mode 0600. Its self-signed certificate is valid from
3 November 2026 to 3 November 2029. The original staged copy is migration
material, not a runtime dependency. Rebuilds must reuse the installed pair through
`SAML_SP_CERT_FILE` and `SAML_SP_KEY_FILE`; back it up privately and coordinate
replacement with the IdP administrator.

Live inspection on 3 November 2026 found IFFLogin's published SAML signing
certificate is self-signed, RSA-4096/SHA-256, valid 28 March 2024–28 March 2027.
Its subject and issuer match and OpenSSL verified its self-signature. That
inspection does not independently authenticate the metadata trust anchor or
establish IFFLogin's policy for SP certificates.

`src/juena/server/auth/sp_metadata.py` exports schema-checked public SP XML
without an IdP fetch or application credentials. It and `SamlService` share
the same SP configuration. The XML contains the public signing certificate,
persistent NameID, POST ACS, POST/Redirect SLO, organization and approved
technical contact. On 3 November 2026 the user requested XML metadata signing.
The offline exporter and application now use PySAML2's signed metadata output
with RSA-SHA256 and SHA-256 digests, using the existing persistent SP keypair.
The signature covers the root `EntityDescriptor` by its unique `ID`. Final
signed bytes are preserved; formatting after signing would invalidate the
signature. The prepared document passed xmlsec verification against the known
SP certificate, and tests reject changed endpoints and a different certificate.
During initial registration, a metadata-only publisher verified the signature
again using xmlsec from the app image with only public inputs and networking
disabled. That bootstrap is complete: the application now serves signed XML
from `/auth/metadata` using the same registered keypair. The former gateway
mounted only public XML, not `sp.key`. An embedded certificate alone does not
establish trust; the SP public key must be accepted independently by IFFLogin.

### 5.4 Internal and browser-facing URLs

The hardcoded backend URL blocker is resolved. Streamlit uses
`JUENA_API_INTERNAL_URL` for pooled server-to-server FastAPI requests and builds
browser login links from `JUENA_PUBLIC_URL`. The internal value is not stored in
chat session controls or rendered in the UI.

Development keeps sensible split-port defaults (`8080` for FastAPI, `9501` for
Streamlit). Production has no localhost public default: startup rejects missing
or non-HTTPS public/SAML URLs and rejects an insecure session cookie.

The base Compose file publishes chat PostgreSQL on loopback for the host
worker. UI, API and RAG ports stay private on the server. Local development
adds [docker-compose.dev.yml](docker-compose.dev.yml), which publishes their
loopback ports and supplies local authentication settings. Use the launcher
to select the correct files:

```bash
./juena up              # local .env and development overlay
sudo ./juena prod up    # installed server environment and selected gateway
```

Do not use the local overlay on the server. Inside the app container, the
database hostname is Compose's `postgres`; the separate host worker reaches
the loopback-published port. `localhost` in a container refers to that container.

---

## 6. Local development without IFFLogin

`AUTH_DEV_BYPASS` provides a local sign-in for a developer machine without a
registered SP. The server's registered SP continues to use real IFFLogin.

```bash
# .env
ENVIRONMENT=development
AUTH_DEV_BYPASS=true
DATABASE_URL=postgresql://juena:juena-development-only@localhost:5432/juena
```

```bash
./juena postgres-up
./juena run             # or use ./juena up for the full Docker stack
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

Keep `AUTH_DEV_BYPASS=false` in the installed server environment.

---

## 7. External confirmations

Daniel has confirmed use of `iff1122.iff.kfa-juelich.de` and described the
nginx TLS/internal-HTTP proxy approach. The user reported that reply on
3 November 2026. The HTTPS certificate/key and dual-host nginx preflight have
passed while the application was initially offline. Its public metadata URL
served the expected XML over trusted HTTPS.
Daniel subsequently required proper SP metadata at its given HTTPS URL.
The XML-signed update was published and independently verified against the
known SP certificate before app activation. During that bootstrap only metadata
was served and login/ACS returned 503. This was a temporary registration step;
the application now serves those routes through the full gateway configuration.

In the access discussion reported on 3 November 2026, Daniel offered manual
activation, PGI/JCNS internal LDAP accounts, FZJ staff, or external guests as
default eligibility options. The user requested JCNS staff first and intends
to invite only several testers. Record that as the requested scope, not as
confirmed JCNS-only filtering: Daniel's listed LDAP option includes both PGI
and JCNS. Exact group scope and applied IdP settings remain to be confirmed.
Sharing the URL with few testers does not enforce a tester-only limit; any
eligible staff member who obtains the URL could sign in once the app is live.
No application allowlist has been added.

Daniel also confirmed SP-initiated login only. Users start at JueNA, whose
`GET /auth/login` creates a signed AuthnRequest; IFFLogin returns the response
to `POST /auth/acs`. JueNA is not listed on the IFFLogin start page. Existing
request correlation and `allow_unsolicited=False` match that agreement.
This login agreement does not settle IdP-initiated Single Logout behavior.

The user subsequently reported Daniel's confirmation that the SP is registered.
Do not keep treating SP registration or acceptance of the submitted metadata as
pending. Validate persistent NameID, optional profile attributes, required
response/assertion signatures and logout behavior during browser acceptance;
resolve actual interoperability failures with IFFLogin if encountered.

JueNA's current production configuration separately requires a pinned certificate
for the IdP metadata signature. Daniel has now confirmed its SHA-256 fingerprint
through the user:
`F8:77:96:4D:CA:47:3E:A5:7D:2C:F5:5D:82:BC:E0:B5:72:43:18:83:33:87:8C:1E:86:76:50:A0:29:A8:11:25`.
The latest published certificate matches that fingerprint. The authenticated
certificate is installed at `/etc/juena/saml/idp-metadata.crt` and the
actual `SamlService` successfully verified fresh IdP metadata and generated a
signed request. The rebuilt application is running; its public API/UI/metadata
return 200 and login returns 303. Use the repository's `juena prod` commands
as described in [the operations guide](deploy/SERVER.md).
This trust setup is separate from the completed SP registration. Future key
rollover and exact JCNS-only versus combined PGI/JCNS eligibility remain to be
documented; no new SP certificate has been requested.

Confirm DNS, TLS termination/renewal, upstream institute/VPN rules and shared
Blablador credentials only with whichever institutional team controls them.
You own Docker, Podman, storage, systemd and the VM firewall on Proxmox.
Any internal administrator correspondence remains in ignored local notes.
The operational steps and acceptance checklist needed for a new deployment
are in this guide and the tracked `deploy/` runbooks.

---

## 8. Acceptance status

- [x] RSA-SHA256 signing and SHA-256 digests, checked on an actual Redirect.
- [x] Verified signed metadata and production trust-certificate requirement.
- [x] Periodic refresh preserving the client state, with expiry rejection.
- [x] Configurable organization and technical contact in SP metadata.
- [x] One HTTPS gateway for internal API/UI routes and public browser URLs.
- [x] Confirm the existing `iff1122` hostname and TLS reverse-proxy approach with Daniel.
- [x] Prepare the authorized persistent SP keypair, offline metadata and approved contact.
- [x] Publish and independently verify the metadata URL through the shared gateway.
- [x] Independently verify inline XML/browser-display response headers.
- [x] Prepare and verify XML-signed SP metadata using the existing keypair.
- [x] Publish the signed update and independently verify its live signature.
- [x] Confirm SP-initiated login only with Daniel; implementation already matches.
- [x] Register the SP with IFFLogin (Daniel's confirmation reported by the user).
- [x] Complete the IdP metadata trust setup and live signature preflight.
- [x] Fix ACS signature policy and PySAML2 response handling; user reported recovery.
- [ ] Confirm exact eligibility and future certificate-rollover procedure.
- [ ] Real browser login, authenticated session, logout and repeat login.
- [ ] Verify IdP-initiated SLO, certificate rollover and metadata refresh on the VM.

Logout verifies Redirect signatures or POST XML signatures before revoking
sessions, correlates responses with pending requests, and can return either a
redirect or a POST form. Actual IFFLogin logout interoperability remains to be recorded.
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
