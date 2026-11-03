# Configure IFFLogin with SAML

Use this workflow when deploying JüNA at a new public hostname or updating an
existing SAML registration. It applies to both standalone and shared gateways.
Examples use `juena.example.org`; replace it with the approved hostname.
Keep real hostnames, administrator correspondence, certificate fingerprints and
acceptance results in private deployment records under ignored `DOCS/`.

See [SERVER.md](SERVER.md) for host preparation and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#ifflogin-returns-401-at-authacs) for ACS
validation failures. Do not use the local authentication bypass on a server.

## 1. Agree the registration details

Confirm with the identity-provider administrator:

- The public SP entity ID and HTTPS metadata URL.
- SP-initiated login, HTTP-Redirect requests and HTTP-POST responses.
- Persistent NameID support and any optional email/display-name attributes.
- Response-signature requirements and the accepted SP signing key.
- POST/Redirect logout bindings and expected IdP-initiated logout behavior.
- Who may sign in, and how eligibility is enforced at the IdP.
- Metadata trust, refresh and certificate-rollover procedures.

Sharing a URL with a few testers does not enforce a tester-only access policy.
Record the agreed scope rather than assuming a particular institute group is
enabled. A registration for one hostname does not automatically cover another.

IFFLogin's metadata URL is configured with `SAML_IDP_METADATA_URL`; the template
uses `https://ifflogin.fz-juelich.de/saml/idp.xml`. Read the freshly verified
document for bindings, signing keys and expiry. Do not deploy from a historical
snapshot or assume a published certificate has not changed.

## 2. Configure the public origin and gateway

Prepare DNS and an HTTPS certificate for the chosen hostname. FastAPI and
Streamlit must share that hostname: the session cookie is host-only with path
`/`, so separate API/UI subdomains do not share it.

Edit `/etc/juena/juena.env`, using the server template on a new installation:

```dotenv
ENVIRONMENT=production
AUTH_DEV_BYPASS=false
JUENA_PUBLIC_URL=https://juena.example.org
UI_PUBLIC_URL=https://juena.example.org
SESSION_COOKIE_SECURE=true
SAML_SP_ENTITY_ID=https://juena.example.org/auth/metadata
SAML_SP_ACS_URL=https://juena.example.org/auth/acs
SAML_SP_SLS_URL=https://juena.example.org/auth/sls
JUENA_SAML_DIR=/etc/juena/saml
SAML_SP_CERT_FILE=/data/saml/sp.crt
SAML_SP_KEY_FILE=/data/saml/sp.key
SAML_IDP_METADATA_URL=https://ifflogin.fz-juelich.de/saml/idp.xml
SAML_IDP_METADATA_CERT_FILE=/data/saml/idp-metadata.crt
SAML_METADATA_REFRESH_SECONDS=21600
```

Set `SAML_ORGANIZATION_NAME`, `SAML_ORGANIZATION_URL`, `SAML_CONTACT_NAME` and
`SAML_CONTACT_EMAIL` to the agreed organization and technical contact.
The `/data/saml` paths are container paths; `JUENA_SAML_DIR` selects the host
directory mounted there read-only.

The shipped nginx configurations send `/auth/*` and other API routes to
FastAPI on port 8080, while Streamlit and `/_stcore/*` use port 9501. Browser
links use the public origin. `JUENA_API_INTERNAL_URL` is for private UI-to-API
requests; its current loopback default works because UI and API run in one
container. It must never become a browser login URL.

## 3. Install signing and trust material

Use separate credentials for these roles:

| Host file | Role |
| --- | --- |
| `/etc/juena/tls/fullchain.pem` and `privkey.pem` | HTTPS certificate chain and private key |
| `/etc/juena/saml/sp.crt` and `sp.key` | Persistent SP signing certificate and private key |
| `/etc/juena/saml/idp-metadata.crt` | Independently authenticated certificate for verifying IdP metadata |

Generate or obtain the SP pair using the key type, validity and trust policy
agreed with the IdP. Preserve it across rebuilds. Keep private keys outside Git
with restricted permissions and secure backups. The public SP certificate is
included in metadata; its presence alone does not establish trust at the IdP.

Obtain the IdP metadata-signing certificate through an independently trusted
administrator channel. Compare its fingerprint with the confirmed value:

```bash
sudo openssl x509 -in /etc/juena/saml/idp-metadata.crt -noout -fingerprint -sha256 -dates
```

Do not establish trust by extracting a certificate from the same unverified
XML it is meant to authenticate. JüNA verifies the metadata signature against
the pinned certificate, then uses keys from that verified document to validate
SAML messages. These roles need not use the same certificate.

## 4. Export and register SP metadata

If registration must precede app startup, use the offline exporter. It shares
SP settings with the runtime and needs neither app credentials, a database nor
an IdP fetch. First build the application image:

```bash
sudo ./juena prod build
```

From the checkout, export with the image's dependencies and persistent SP pair.
Replace the example identity/contact values and use the tag configured by
`JUENA_IMAGE` if it differs from the server template:

```bash
sudo docker run --rm --network=none --read-only --tmpfs /tmp \
  --mount type=bind,src="$PWD/src/juena/server/auth/sp_metadata.py",dst=/sp_metadata.py,readonly \
  --mount type=bind,src=/etc/juena/saml,dst=/saml,readonly \
  --entrypoint /app/.venv/bin/python juena-chatbot:dev-server \
  /sp_metadata.py \
  --entity-id https://juena.example.org/auth/metadata \
  --acs-url https://juena.example.org/auth/acs \
  --sls-url https://juena.example.org/auth/sls \
  --cert-file /saml/sp.crt --key-file /saml/sp.key \
  --organization-name 'Example Institute' \
  --organization-url https://institute.example.org \
  --contact-name 'Technical Contact' --contact-email admin@example.org \
  > /tmp/juena-sp-metadata.xml
```

Check that the command succeeds, then verify the exported signature against
the known SP certificate using `xmlsec1` from the application image:

```bash
sudo docker run --rm --network=none --read-only --tmpfs /tmp \
  --mount type=bind,src=/tmp/juena-sp-metadata.xml,dst=/metadata.xml,readonly \
  --mount type=bind,src=/etc/juena/saml/sp.crt,dst=/sp.crt,readonly \
  --entrypoint xmlsec1 juena-chatbot:dev-server \
  --verify --pubkey-cert-pem /sp.crt --trusted-pem /sp.crt \
  --id-attr:ID urn:oasis:names:tc:SAML:2.0:metadata:EntityDescriptor \
  --enabled-reference-uris same-doc /metadata.xml
```

Metadata includes persistent NameID, POST ACS, POST/Redirect SLO, the public
SP certificate and contact details. It is signed with RSA-SHA256 and SHA-256
digests. Do not reformat signed XML: whitespace changes can invalidate it.

Supply the verified XML and authenticate the SP public key through the agreed
registration channel. If the IdP requires the HTTPS URL before activation,
serve only the verified XML at `/auth/metadata` using a temporary HTTPS
publisher. The [shared-gateway guide](SHARED-PROXY.md#metadata-before-app-startup)
describes the provided metadata-only overlay; on a dedicated host use an
equivalent temporary HTTPS route. Mount only public XML into the publisher,
not the SAML private key. Confirm registration before expecting login to work.

## 5. Deploy and test login

Once production settings, IdP trust and registration are ready:

```bash
sudo ./juena prod config
sudo ./juena prod up
sudo ./juena prod health
```

On a dedicated server, `PROD_PROXY_MODE=standalone` explicitly selects its
gateway. Shared mode uses the installed independent gateway. Neither mode
changes the SAML configuration or response validation.

The app serves signed metadata at `GET /auth/metadata`. Download it from the
public origin, compare entity/endpoints/certificate with the registered values,
and verify its signature. Start a fresh login from JüNA; do not replay an old
ACS form after a restart or failed attempt.

| Endpoint | Purpose |
| --- | --- |
| `GET /auth/login` | Start a signed, SP-initiated authentication request |
| `POST /auth/acs` | Validate the response and create the app session |
| `GET /auth/metadata` | Publish signed SP metadata |
| `POST /auth/logout` | Start logout from an authenticated app session |
| `GET` / `POST /auth/sls` | Process SAML logout requests and responses |

Point `SAML_SP_SLS_URL` at `/auth/sls`, not `/auth/logout`. The IdP callback may
arrive without an app cookie and is authorized by SAML signature validation.

IFFLogin's logout status page uses background browser requests. JüNA therefore
allows the origin of `SAML_IDP_METADATA_URL` to read `/auth/sls` responses via
CORS, including redirects and validation failures. This permission applies only
to that endpoint, for GET/POST and their preflight requests. It does not grant
the IdP browser access to profile, chat or file APIs. The SAML signature is still
required; the app session cookie keeps its existing SameSite policy. The fix is
in the application and applies to both gateway modes.

## Validation policy and compatibility fixes

JüNA requires a signed Response. A separate Assertion signature is optional
and verified when present; the enclosing Response signature covers its
assertion. Metadata advertises `WantAssertionsSigned="false"`. Do not disable
response checking to work around `Signature missing for assertion`.

The callback uses PySAML2's validated response/session data rather than calling
the nonexistent `AuthnResponse.is_authenticated()` in PySAML2 7.5.4. Validation
errors remain authentication failures. Request correlation, destination,
audience and persistent NameID checks remain enforced; unsolicited responses
are rejected. Optional profile attributes do not replace the stable NameID.

The local authentication bypass is only for developer machines. It skips SAML
initialization, so SAML endpoints return 503 while enabled. Production rejects
`AUTH_DEV_BYPASS=true` at startup.

## Acceptance checklist

Record results for each installation in private deployment notes:

- [ ] DNS, TLS chain and public routes work from the intended client network.
- [ ] SP metadata endpoints/key match registration and its signature verifies.
- [ ] The IdP metadata trust certificate is independently confirmed and installed.
- [ ] The actual access scope and optional attribute names are agreed and tested.
- [ ] A fresh login creates the expected stable account and app session.
- [ ] Logout revokes the session; repeat login retains the same account/files.
- [ ] IdP-initiated logout and agreed POST/Redirect bindings work end to end.
- [ ] Logout after an app restart is checked with the IdP administrator.
- [ ] Metadata refresh, expiry handling and certificate rollover are exercised.

A health response or 303 redirect alone is not browser acceptance. The PySAML2
request/logout cache is in memory; persisted local sessions do not preserve its
IdP session index across process restarts.

## Refresh, renewal and hostname changes

Production fetches fresh HTTPS metadata at startup and verifies its signature.
Refresh defaults to six hours, earlier near expiry, while preserving client
and pending request state. A failed refresh retains verified metadata only
until `validUntil`; expired metadata blocks new SAML operations with 503.
There is no stale disk fallback at startup.

Coordinate metadata trust-anchor replacement with the IdP administrator.
Assertion-key rollover is learned through verified metadata. Coordinate SP
key replacement with registration updates; ordinary TLS renewal is separate.

For a new hostname, update both public app URLs and all `SAML_SP_*` entity/
endpoint URLs, install matching TLS material, export/register revised metadata,
redeploy and repeat the checklist. Use the same workflow when moving from a
shared gateway to a dedicated server.
