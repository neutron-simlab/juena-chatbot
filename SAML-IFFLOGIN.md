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

## 2. What the IdP actually supports

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

Single Log-Out is advertised on both bindings, so Daniel's stated advantage is
real and usable.

> **The values above are a snapshot taken on 5 August 2026 and the metadata
> expires on 12 August 2026.** Two consequences. First, re-read the live
> document before relying on any row in this table — a refreshed `idp.xml`
> carries a new `validUntil` and may carry new certificates. Second, the roughly
> one-week validity window is not a quirk of this snapshot but the IdP's normal
> cadence, which makes §4.3 a release blocker rather than a nicety: a deployment
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

Endpoint layout ([auth_endpoints.py](src/juena/server/auth_endpoints.py)):

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

## 4. Gaps found while verifying

Four issues in our SP configuration, all confirmed against the installed
`pysaml2==7.5.4`.

### 4.1 We would sign with SHA-1

PySAML2 defaults to `rsa-sha1` signatures and `sha1` digests. The IdP
advertises RSA-SHA256 and above. SHA-1 is deprecated and Shibboleth-family
IdPs commonly reject it outright.

**Fix:** set `signing_algorithm` and `digest_algorithm` to the SHA-256 variants
in the SP config. Both are valid PySAML2 configuration keys.

### 4.2 The IdP metadata signature is never verified

The metadata is signed, but PySAML2's `parse_and_check_signature` begins with
`if not self.cert: return True` — with no certificate configured, verification
is **silently skipped**. Our config passes only `{"url": ...}`, so trust rests
entirely on TLS.

**Fix:** obtain the metadata signing certificate from Daniel and pass it as
`cert` alongside the URL.

### 4.3 Metadata expiry will break logins with no deploy to explain it

`validUntil` sits about a week out, PySAML2 defaults to `check_validity=True`
and **drops expired entities**, and `saml_lifespan` fetches the metadata
**once at startup**. A container running past that timestamp stops recognising
the IdP entirely and every login fails.

**Fix:** periodic reload, or an MDQ endpoint if IFFLogin offers one.

### 4.4 Attribute names are guesses

The IdP metadata declares no attributes, so nothing confirms the wire names.
`_first_attribute` in [saml_service.py](src/juena/server/saml_service.py)
currently tries three spellings each for e-mail and display name.

**Fix:** ask Daniel for the exact released attribute names.

### 4.5 Our SP metadata omits organisation and contacts

The IdP publishes `Organization` and two `ContactPerson` entries; federations
usually expect the same in return. Our `SPConfig` sets neither, though PySAML2
supports `organization` and `contact_person`.

---

## 5. Deployment prerequisites

### 5.1 The SP base URL

`https://juena.fz-juelich.de` — the **public HTTPS origin where FastAPI is
reachable from the user's browser**, not from inside Docker. SAML redirects the
browser, so the address must resolve on the user's machine.

Daniel is expected to provision the hostname and its TLS certificate. The
production settings then are:

```bash
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
> and Streamlit share one container**, which is the current layout.
> [DEPLOYMENT-PLAN.md](DEPLOYMENT-PLAN.md) splits them
> into separate `api` and `ui` services, where `localhost` inside the UI
> container is the UI itself and the value becomes `http://api:8080`. Change it
> in the same release as the split; nothing else in this file is affected,
> because every other URL here is browser-facing.

### 5.2 Both services must share one hostname

`/auth/acs` sets `juena_session` with `path="/"` and **no `domain=`**
([`_set_session_cookie`](src/juena/server/auth_endpoints.py#L53-L62)), so the
cookie is host-only. Streamlit then reads it through `st.context.cookies`.

Splitting the two across subdomains — FastAPI on `api.…`, Streamlit on
`chat.…` — means the cookie is never sent to Streamlit, and login fails
silently: the user round-trips through the IdP and lands back logged out. It
works locally only because `localhost:8080` and `localhost:9501` are the same
host; cookies ignore the port.

So both run behind `juena.fz-juelich.de`, split by path in the reverse proxy:

| Path | Backend |
| --- | --- |
| `/auth/*`, `/chats/*`, `/agents`, `/repositories`, `/health`, `/{agent_id}/*` | FastAPI `:8080` |
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

## 7. Questions outstanding for Daniel

1. **Signing algorithm** — confirm RSA-SHA256 + SHA-256 digest; does the IdP
   reject SHA-1?
2. **Metadata signing certificate** — cert or URL, so we can verify the signed
   `idp.xml`.
3. **Attribute names** — exact wire names released for e-mail and display name.
4. **Persistent NameID** — released by default, or enabled per SP?
5. **Metadata refresh** — expected polling cadence, or is there an MDQ endpoint?
6. **Bindings** — confirm Redirect-out / POST-back for SSO and both bindings for
   SLO are acceptable.
7. **Registration process** — how to submit SP metadata on his side.
8. **SP signing keypair** — does he issue one for `juena.fz-juelich.de`, or
   should we generate a long-lived self-signed keypair and publish the public
   half in our metadata? (See §5.3 — distinct from the TLS certificate.)
9. **Hostname and TLS** — confirm `juena.fz-juelich.de` is what will be
   provisioned, and that both services can sit behind it on one origin (§5.2).

---

## 8. Next steps

**Blocked on Daniel:** items 3 and 4 — attribute names and persistent NameID
behaviour — plus item 8, which decides where the SP keypair comes from. Nothing
else in the list needs his reply to start.

**Blocked on infrastructure:** provisioning of `juena.fz-juelich.de` with TLS,
and a reverse proxy routing both services onto that single origin (§5.2).

**Can be done now:**

- [ ] Pin SHA-256 signing and digest algorithms (§4.1)
- [ ] Support a metadata signing certificate in the SP config (§4.2)
- [ ] Add periodic IdP metadata refresh (§4.3)
- [ ] Add `organization` and `contact_person` to the SP metadata (§4.5)
- [x] Make the Streamlit backend URL configurable, internal vs public (§5.4)

Once those land and the hostname exists, `GET /auth/metadata` produces the
final document to hand over, and staged live login QA can begin.

---

## 9. References

- IdP metadata: <https://ifflogin.fz-juelich.de/saml/idp.xml>
- PySAML2 configuration: <https://pysaml2.readthedocs.io/en/latest/howto/config.html>
- SP implementation: [saml_service.py](src/juena/server/saml_service.py),
  [auth_endpoints.py](src/juena/server/auth_endpoints.py)
- Settings: [config.py](src/juena/core/config.py), [env.example](env.example)
