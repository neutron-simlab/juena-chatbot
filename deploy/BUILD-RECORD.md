# Image build record

Build and deployment evidence, newest first. Each entry states whether only
the build passed or deployment/runtime acceptance was completed. Historical
entries describe the state at their date; use the newest entry for pilot status.

## Current build provenance

Build from the chatbot checkout alone. `pyproject.toml` and `uv.lock` pin
published Git revisions of `juena-core` and `juena-rag`; the Docker build supplies
Node.js 24 for core's packaged frontend. Sibling repositories are not required.
The older entries below describe the source layout used at their build dates.

Record the chatbot commit and any modified build inputs, the dependency pins,
lockfile hash, local image ID and resolved base-image digests. The image ID is
not a registry manifest digest. Preserve the reviewed image for deployment and
rollback; a successful build alone does not prove application/SAML readiness.

```bash
git status --short
git rev-parse HEAD
sha256sum uv.lock
# Use the reviewed production environment; this command builds only the app.
sudo ./juena prod build
sudo docker image inspect juena-chatbot:dev-server --format '{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}'
```

---

## 2026-11-03 — auth, retrieval and sandbox recovery

The fixes and deployment documentation are maintained on
`fix/portable-production-auth-sandbox`, based on chatbot commit
`cce102ed1ba2f1ad15d0b492f5db369336a6511a`. The implementation is recorded in
commit `1a18ba4`; the pilot had been deployed with local changes before that
commit. Do not identify its image with the older image ID recorded below;
record the final commit and running image ID at the next deployment.
Core/RAG pins remain `0e90214c1da91df10845e5a0c425f1c7c3394a3c` and
`db98edd752967a48ef49fd61f7ee53ad47bcc23c`.

| Area | Evidence and status |
| --- | --- |
| Host capacity | User increased the VM allocation to 12 GB; a subsequent host check showed about 11 GiB total RAM. Container limits and one sandbox slot remain conservative. |
| IFFLogin | SP registration and metadata trust were confirmed. ACS failed with `Signature missing for assertion`; the signed-Response policy and PySAML2 callback were fixed. The user subsequently reported the issue resolved. The complete browser acceptance sequence remains unrecorded. |
| Repository discovery | Initial diagnostics found jscatter indexed but no configured/cloned repositories. After the application-root/registry-path fix, diagnostics loaded all eight IDs and found all eight local clones. |
| Retrieval contents | The user's 18:03 UTC diagnostic reported jscatter with 1,676 chunks. Ingestion of datreat, mdanse, sasview, sasmodels, sasdata, silx and vitess was subsequently started. Later output showed tokenizer download completion and an embedding batch of 12,361 chunks; final counts for the seven are not yet recorded. |
| Sandbox | The user's 18:03 UTC diagnostic reported `enabled=true`, `jobs_table_exists=true`, and `worker_connected=true`, with Podman 4.9.3 and the dedicated account installed. This establishes the reported app/queue/connection state, not browser delivery of a plot. |
| Gateway | The shared gateway serves JueNA alongside eLabFTW. The launcher also supports standalone mode; SAML, repository and worker fixes are independent of gateway selection. |

Worker setup fixes address the inherited private checkout directory, traversal
of `/var/lib/juena-sandbox` by remapped UIDs, and `runc 1.2.2` failing to enter
app-owned scratch directories. Setup now starts account commands from `/`, sets
only the parent to the explicitly approved `0711`, and selects `crun` for the
worker and generated jobs. It retains restricted workspace permissions, imports
the app's image, waits for readiness, and gates enablement on a queued PNG test
before and after redeployment. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Regression coverage includes signed SAML responses and alternate public origins,
repository paths and tool availability, sandbox preflight/activation/rollback,
and launcher lifecycle in both gateway modes. The branch review ran 154 passing
tests in an isolated Python 3.12 environment using the pinned core source;
three pgvector integration tests were skipped because no disposable test DSN
was configured. Shell syntax, local documentation links and both Compose modes
using the public example environment were also checked.
These local tests do not substitute for a root lifecycle run or browser
acceptance on another host.

Remaining evidence to record:

- All eight repositories listed with nonzero counts after ingestion completes.
- A PNG generated and delivered through a new authenticated chat, plus Files
  panel preview/download through the public gateway.
- Signed-in identity, logout, repeat login, and IdP-initiated SLO behavior.
- Reboot recovery, backup restoration and the broader sandbox acceptance checks
  in [SANDBOX.md](SANDBOX.md#broader-rollout-acceptance).
- Final chatbot commit, running app/worker image IDs and dedicated-server runtime
  acceptance when that host is provisioned.

## Historical records

The entries below retain earlier build and migration evidence. Statements such
as "worker disabled" or "registration pending" describe those earlier stages,
not the current pilot. Former `.sandbox/server-setup` helpers and sibling-source
build layouts are not required by today's launcher.

## 2026-11-03 — application activation and native CLI

The initial installer completed while the launcher was being updated. Public
HTTPS checks returned API/UI health 200, metadata 200, login 303 and eLabFTW 302.
The running image ID is recorded by the installer under its protected backup;
it has not been independently read in this session. The old image ID below
must not be treated as the current running image. Browser login/logout remains
to be accepted.

Ongoing deployment uses `sudo ./juena prod up` from the working source checkout.
It builds that checkout with the protected installed environment; no source
copy or temporary installer is involved. The shared Compose overlay and gateway
templates are now in the repository. The initial `/opt` checkout is retained as
an installation snapshot. The CLI update was checked with launcher tests and
the real Compose model; a root lifecycle run through the updated CLI remains
to be exercised by the operator.

## 2026-11-03 — server image built from the chatbot checkout

**Build passed; deployment/runtime acceptance pending.** The user supplied the
completed Compose build output: all 27 steps finished, frozen dependency installs
succeeded, and the runtime image was exported under the tag below. This assistant
session cannot independently inspect the Docker daemon because sudo still
requires a terminal password.

| Field | Recorded value |
| --- | --- |
| Image tag | `juena-chatbot:dev-server` |
| Image ID from build output | `sha256:03048a277102fe0a8a0c8a678931c48ee9f15165db7c2e38c571fbec809bc8e3` |
| Chatbot checkout HEAD | `cce102ed1ba2f1ad15d0b492f5db369336a6511a` |
| Core Git pin | `0e90214c1da91df10845e5a0c425f1c7c3394a3c` |
| RAG Git pin | `db98edd752967a48ef49fd61f7ee53ad47bcc23c` |
| `uv.lock` SHA-256 | `7cfbb09879cbb35ad627bb6cb729f1b7f4fcf9b428bae6a6405617d551521e54` |
| Runtime/build Python base | `python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd` |
| Frontend Node base | `node:24-bookworm-slim@sha256:c86ecead9e55a5210ac1727de38a4ca840bf03dcaf34dc9749addbf383b1920c` |
| Build-tool image | `ghcr.io/astral-sh/uv:0.7.15@sha256:9653efd4380d5a0e5511e337dcfc3b8ba5bc4e6ea7fa3be7716598261d5503fa` |
| Source of evidence | User-provided terminal build output, 2026-11-03 |

At build time, the checkout had deployment documentation/template/nginx-comment
edits under `deploy/`; those files are outside the application image's
allowlisted inputs. Application build inputs then matched the recorded commit.
Base-image identifiers above are the digests shown in that build output.

**Subsequent source change, not in this image:** metadata registration preparation
on 3 November extracted the SP configuration into
`src/juena/server/auth/sp_metadata.py`, shared by the runtime and a standalone
offline exporter. Rebuild the application image from the reviewed current
checkout before starting it, and record the new image ID. Metadata-only nginx
publication serves the already-generated public XML and does not run this image.
Subsequent metadata work enables XML signatures with the existing SP key and
inline XML response headers. The publisher uses only the existing image's
`xmlsec1` binary for an isolated public-input signature check, without starting
the application or requiring that image to contain the new source code.

The user built from `/home/administrator/juena-chatbot` using the private staged
environment with `docker compose -f docker-compose.yml --env-file
.sandbox/server-setup/juena.env build juena-chatbot`. This creates the image only.
Application startup, installed package/frontend inspection, database setup and
real SAML/browser acceptance have not been established by this build. IFFLogin
registration/trust/SP key policy and remaining runtime settings are still pending.

---

## 2026-09-16 — the client/API runtime proof, with the subagent execution fix

The image the client/API portion of the four step-4 conversations was proved
against. Same build shape as the entry below; what changed is core. This record
does not claim the browser-only rendering checks: those remain an acceptance
gate in plan 02.

| | |
|---|---|
| **Image ID** | `sha256:19434757dcfd3440d7e36e432a892c808819816d3273fda48082cc5347adca0b` |
| **juena-chatbot** | `0767d3913a57f6e92807338a6179f68cfe9d3dc3`, tree clean |
| **juena-core** | `a64c978a7d5cb03c3dfebeabddfb29ae7f10dbb3`, tree clean |
| **juena-rag** | `b7b72954477e781c1bda0828e24c4ecf0ffbccd0`, pinned by `uv.lock` |
| **Resolved set** | 160 packages; `uv.lock` sha256 `71b114dce71a2dfcb3234afe17078a8e30da6ee7020c276503188549511c6a7d` — unchanged from the previous entry |
| **Credentials used** | none |

Two shipped sources changed relative to the previous entry. Core fixes sandbox
`execute`, which raised `RuntimeError: Sandbox execution requires a graph
run_id` because the id it looked for does not exist inside a subagent — the only
place that tool is bound. The application changes
`src/juena/research/runner.py` so a background run supplies its own invocation
id. The intervening `./juena` and `tests/` changes do not ship in the image.
See the plan's Step-4 record. Rebuilding from the clean commits above reproduced
this exact image ID.

---

## 2026-09-16 — the juena-core cutover, first parent-context build

The first image built after juena-chatbot stopped carrying the shared
infrastructure and started consuming `juena-core` (plan 02, step 3.5).

| | |
|---|---|
| **Image ID** | `sha256:352c43f42b5f8638523de87f5c73677dc2cd14cbc31ff6f94570ef6422f9fd10` |
| **juena-chatbot** | `2a9c1d785106b23855dceea53d9538fb757f1c7d`, tree clean |
| **juena-core** | `4eca16e99afba03023fadc22a8a9a37e2a43c9b4`, tree clean |
| **juena-rag** | `b7b72954477e781c1bda0828e24c4ecf0ffbccd0`, pinned by `uv.lock` |
| **Resolved set** | 160 packages; `uv.lock` sha256 `71b114dce71a2dfcb3234afe17078a8e30da6ee7020c276503188549511c6a7d` |
| **Base image** | `python:3.11-slim` |
| **Built by** | `./juena build`, which is `docker compose build` over the two development Compose files |

**Credentials used: none.** `juena-core` is copied from the build context, and
`juena-rag` is cloned anonymously over public HTTPS. Nothing in the build reads
a token, an SSH key or a registry login.

### What changed in this build, and why

The build context — the set of files Docker is allowed to read — moved from this
repository to its **parent directory**. A Dockerfile can only copy files from
inside its context, and `juena-core` sits *beside* this repository rather than
inside it, so with the old `context: .` there was no way to reach it.

Three consequences, all of them visible in the files:

1. **Every `COPY` in the `Dockerfile` now starts with a repository name** —
   `COPY juena-chatbot/src/ ./src/` rather than `COPY src/ ./src/`.
2. **Core lands at `/juena-core`, not under `/app`.** uv reads the path
   `../juena-core` relative to the project directory, which is `/app` inside the
   image; one level up from `/app` is the filesystem root. uv builds a wheel
   from that directory during `uv sync`; nothing imports it from there at
   runtime, and the installed copy under `/app/.venv` is what the application
   loads.
3. **`.dockerignore` is no longer consulted.** Docker looks for an ignore file
   named after the Dockerfile first, and only falls back to one at the context
   root — and the context root is now the parent directory, which has none. So
   `Dockerfile.dockerignore` is the only filter in effect. This is the part
   worth checking on any change: the parent directory holds about nine
   gigabytes across a dozen unrelated repositories, several carrying real
   credentials in their own `.env`. The ignore file therefore excludes
   everything and re-admits only `juena-chatbot/` and `juena-core/`.

### What was verified

```text
build context transferred            2.79 MB   (unfiltered parent: ~9 GB)
.env / *.key / *.pem in context      none
.git / .venv in context              none
prompt files in context              SUPERVISOR.md + both AGENT.md present
image: juena_core resolves to        /app/.venv/lib/python3.11/site-packages/juena_core
image: juena resolves to             /app/src/juena
image: /app/.env                     absent
image: /app/tests                    absent
rebuild from the clean commit        same image ID
Podman socket in resolved Compose    none, in either service
```

The context was measured by building a throwaway image that copied the whole
context and listed it, rather than by reading the ignore file and believing it.
An ignore rule that silently fails to match looks exactly like one that works.
