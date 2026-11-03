# Deployment troubleshooting

These checks apply to both the shared gateway and a dedicated server. Run them
from the checkout used for deployment with the installed production environment:

```bash
sudo ./juena prod ps -a
sudo ./juena prod health
sudo ./juena prod doctor
sudo ./juena prod logs --tail 100 juena-chatbot
sudo ./juena prod logs --tail 100 juena-rag
sudo ./juena prod sandbox-logs
```

Application/RAG logs follow until Ctrl+C; the worker journal prints full lines
without a pager. `rag-init` exiting with status 0 is expected after schema
initialization. A healthy container proves its health check passed, not that
login, all repository searches, or plotting work. When sharing logs, omit
credentials, cookies, raw SAML messages and authentication query strings.

The authoritative server settings are in `/etc/juena/juena.env`. Editing the
checkout's `.env` affects local development. Use `sudo ./juena prod up` after
app/settings changes; it builds from the checkout containing the launcher.
For an explicitly selected gateway, prefix commands with
`sudo env PROD_PROXY_MODE=standalone` or `sudo env PROD_PROXY_MODE=shared`
as described in [SERVER.md](SERVER.md).

## IFFLogin returns 401 at `/auth/acs`

**Observed:** the browser returned from IFFLogin to
`{"detail":"SAML response validation failed"}`. Server logs showed
`Signature missing for assertion`.

**Cause and fix:** the SP required signatures on both the Response and its
Assertion. IFFLogin supplied a signed Response with an assertion covered by that
signature. The shared SP configuration now requires the Response signature and
makes the additional Assertion signature optional, verifying it when present.
It advertises `WantAssertionsSigned="false"` in metadata. The registered entity,
endpoints and SP keypair stay the same.

The callback also no longer calls `is_authenticated()` on PySAML2 7.5.4's
`AuthnResponse`; that method does not exist. Parsing and validated session-data
extraction are handled together. Failed validation remains an authentication
error. Signature validation, request correlation, destination, audience and
persistent NameID checks remain enforced.

Deploy the fixed checkout with `sudo ./juena prod up`, then begin a **new** login
from JueNA. Do not replay an old ACS form: it belongs to an earlier request.
Verify the signed-in identity, logout and repeat login using
[the SAML checklist](SAML-IFFLOGIN.md#acceptance-checklist). A 303 login redirect
alone is not an end-to-end login test.

This is an application/IdP compatibility fix. On a dedicated host, configure the
new public origin, SAML URLs and TLS certificate and coordinate updated metadata
with IFFLogin; no shared-nginx-specific authentication code is required.

## IFFLogin says to sign out manually after JüNA logout

First open JüNA in a new tab without signing in again. A sign-in prompt confirms
the local app session was revoked; the IdP status page can still report a failed
SAML logout exchange.

IFFLogin performs logout from its status page using background browser requests
([implementation paper, §3.4](https://pgi-jcns.fz-juelich.de/pub/doc/Seminararbeiten/Seminararbeit_MaximilianHeuwes.pdf)).
The browser must be allowed to read the `/auth/sls` reply before following its
redirect back to the IdP. A 303 in server logs alone does not prove this worked:
without `Access-Control-Allow-Origin`, the browser can block the exchange after
the app has already revoked the session.

The application now permits the exact configured IdP metadata origin on
`/auth/sls` only. It handles GET/POST preflights and includes CORS headers on
signed replies and validation errors. Do not add the IdP to the global API
origin list, use wildcard origins, or disable signature checking.

Deploy with `sudo ./juena prod up`, then start a fresh login before testing
logout, since deployment replaces the in-memory SAML request/session cache.
Check both the local sign-in prompt and the IdP's logout status. If `/auth/sls`
returns 401, inspect the safe `SAML logout request rejected (...)` or
`SAML logout response rejected (...)` diagnostic. A 401 is a validation or
correlation failure and needs investigation separately from CORS. Share those
labels and status codes, not raw SAML query strings.

## RAG is healthy but repositories or search tools are missing

**Symptom:** the service has indexed chunks, while the chatbot reports
no configured/cloned repositories. Its warning names a registry path under
`site-packages/juena/core/config/`.

**Cause and fix:** an installed wheel was treated as a source checkout when
resolving application files. The image now sets `JUENA_APP_ROOT=/app` and Compose
sets `REPO_CONFIG_PATH=/app/config/repositories.yaml`. Native source runs still
discover their checkout. Rebuild with `sudo ./juena prod up`; existing indexed
chunks do not need re-ingestion for this path fix.

The agent also keeps repository catalogue/search tools available when repositories
are configured but the index is empty or temporarily unavailable. A later ingest
can then be discovered without permanently losing those tools from the cached
agent graph.

Interpret `prod doctor` as follows:

| Field | What it establishes |
| --- | --- |
| `configured` | IDs loaded from the application's registry |
| `reachable` | The RAG health request succeeded |
| `indexed` | The RAG catalogue and chunk counts; compare every desired ID |
| `cloned` | Local working trees are available for file reading |
| `needs_ingestion` | True when no indexed repository has any chunks; false does not prove complete coverage |

Cloning all eight repositories does not index them. To add the seven other than
the initially ingested jscatter:

```bash
sudo ./juena prod repos-ingest datreat mdanse sasview sasmodels sasdata silx vitess
sudo ./juena prod doctor
```

Omit IDs to process the whole registry. Runs process repositories sequentially
and skip unchanged revisions. If a run fails, inspect its final errors, retry
the affected IDs, and check their catalogue counts. Do not delete the database
or re-ingest successful entries with `--force` merely to retry failed entries.
See [HOW-TO-ADD-REPO.md](../HOW-TO-ADD-REPO.md) for configuration changes.

### Hugging Face warning during Blablador ingestion

`tokenizer.json` is downloaded for **local token counting** before embedding.
Repository text is counted locally and embedding batches are sent to the
configured Blablador API. The embedding model is not downloaded or run locally,
and token counting does not send repository text to Hugging Face.

An anonymous HF Hub rate-limit warning is not a Blablador authentication failure.
If the download completed and output says
`embedding ... with alias-qwen3-8b-embeddings`, let the run continue. No HF token
is required for that successful download. Repeated downloads can occur in fresh
ingestion containers because the Compose stack does not persist an HF cache.

## A plot script exists, but no PNG was generated

Saving `/workspace/plot_sine.py` or seeing files in the Files panel proves file
access, not code execution. The panel works even when execution is disabled.

Check `sandbox` in `prod doctor`:

| Result | Next step |
| --- | --- |
| `enabled=false` | Complete `sudo ./juena prod sandbox-setup`; do not enable the flag alone |
| `jobs_table_exists=false` | Setup creates the optional job table before testing the worker |
| `worker_connected=false` | Inspect `prod sandbox-logs`; setup may have stopped the worker after a failed test |
| All three are true | Run `sudo ./juena prod sandbox-test` to verify actual queued execution and PNG output |

`worker_connected` observes a database advisory lock. The PNG test separately
checks execution, output bytes and artifact reporting. After a successful
deployment, refresh the browser and retry the plot in a new conversation,
approve execution, and check the delivered image. If that still fails, inspect
both app and worker errors; an enabled switch alone is not evidence of success.

## Rootless worker setup failures encountered

### `cannot chdir to /home/operator/juena-chatbot: Permission denied`

The dedicated worker inherited the administrator's private checkout as its
working directory. The helper now changes to `/` before switching accounts.
It builds with Docker from the checkout, imports the images into the worker's
Podman storage, and runs the worker without needing access to the administrator
home. Retry `sudo ./juena prod sandbox-setup` from the fixed checkout.

### `error preparing rootfs` / `remount-private ... permission denied`

With `keep-id`, container root maps to a subordinate host UID. It cannot bypass
permissions on a host-root-owned ancestor. The original `0710` mode on
`/var/lib/juena-sandbox` blocked traversal needed during container mount setup.

Setup now sets **only that parent** to `0711` (`rwx--x--x`): the owner can
list/write/traverse, while other users can traverse a known path but cannot list
or write the parent. Child permissions still decide access to their contents.
The workspace root stays `2770`, and workspace files normally use `0660`.
Do not apply `0711` recursively or make conversation files world-readable.
The preflight tests the real rootless mount with read-only workspace access
before starting the worker.

### `mkdir /workspace/.scratch: permission denied`

The app creates some workspace directories as host root with the worker group.
The observed `runc 1.2.2` checks the working directory before adopting the job's
UID/GID and fails on these restricted directories. `crun` retries access after
switching to the configured execution user.

Setup installs `crun` even when Podman is already installed, uses
`--runtime=crun` for the worker and preflight, and writes
`SANDBOX_OCI_RUNTIME=crun` for generated jobs. It restarts only the dedicated
account's Podman API so a previously running API discovers the new runtime.
Workspace permissions stay restricted; global engine settings are unchanged.

### `sandbox worker did not become ready`

Read complete errors with `sudo ./juena prod sandbox-logs`. Setup stops and
disables the worker after readiness or smoke-test failure. Partial installation
does not mean execution is enabled; correct the reported error and rerun
`sudo ./juena prod sandbox-setup`. It can reuse the account/images and backs up
existing protected configuration. Success tests a queued PNG, enables the app,
redeploys, and tests again. See [the installation sequence](SERVER.md#install-the-worker).

## Gateway and lifecycle checks

`PROD_PROXY_MODE=auto` chooses shared mode when `/etc/juena-edge` exists and
standalone mode otherwise. An explicit mode is a shell option, not an app env
setting. A dedicated VM does not need the shared overlay or `juena-edge` project.

`prod up` refreshes the app and synchronizes the installed worker image. Changes
to the worker unit, execution image or worker settings require `prod sandbox-setup`
again. `prod down` stops the worker before its database and keeps data volumes.
In shared mode it leaves eLabFTW and the gateway running; JueNA's hostname can
return an upstream error while the app is stopped. See
[shared-gateway recovery](SHARED-PROXY.md) for its independent nginx project.
