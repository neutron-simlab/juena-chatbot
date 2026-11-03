# Linux development server

This deployment targets a full Linux VM on Proxmox, Docker Engine with Compose
2.20+ (or Compose 5), and a separate host worker using rootless Podman. Only the
chatbot repository is cloned. `uv.lock` fixes the published core/RAG revisions;
Docker builds core's Files frontend with Node.js 24.

```text
Browser (institute/VPN) → gateway :443 → chatbot :9501 (UI), :8080 (API/SAML)
                                          ├→ postgres :5432
                                          └→ juena-rag :8100 → rag-db :5432
Host worker → 127.0.0.1:5432 → sandbox_jobs → rootless Podman jobs
```

All Compose services share a private project network. RAG, UI and API have no
published server ports. Chat Postgres is published on loopback for the worker.
The development overlay publishes loopback UI/API/RAG ports; do not load it on
the shared server. No Docker or Podman socket is mounted into the application.

## Prepare the VM

Install Docker Engine, its Compose plugin, Git, Python 3.11+, uv, Node.js 24/npm,
rootless Podman and its distribution-specific uidmap/network/storage tools.
Node/npm are needed for the host worker's source install, and live only in the
build stage of the application image. Use cgroup v2 with memory/CPU delegation.
Use a dedicated unprivileged `juena-sandbox` account with subordinate UID/GID
ranges and lingering. It does not need membership in the Docker group.

As administrator, place the checkout at `/opt/juena/juena-chatbot` and create:

```bash
install -d -m 0700 /etc/juena /etc/juena/saml /etc/juena/tls
install -d -o juena-sandbox -g juena-sandbox -m 2770 /var/lib/juena-sandbox/workspaces
install -m 0600 deploy/server.env.example /etc/juena/juena.env
```

Edit `/etc/juena/juena.env`. Replace all `CHANGE_ME` values. Use distinct random
hex passwords (`openssl rand -hex 32`) for the databases and a separate stable
filesystem identity secret. These passwords appear in connection URLs, so hex
avoids URI escaping mistakes. Keep the identity secret and workspace path stable
across upgrades. Never use the development bypass on this server.

Place the HTTPS full chain and private key at `/etc/juena/tls/fullchain.pem` and
`privkey.pem`. The supplied gateway terminates TLS itself. If TLS must terminate
at an institutional proxy, agree that topology first and adapt the gateway and
forwarded-header trust; this template does not configure that alternative.

Use a separate, persistent SAML SP keypair in `/etc/juena/saml/sp.key` and `sp.crt`.
Generate it only after confirming the accepted lifetime and rollover procedure
with IFFLogin. The SP public certificate is included in `/auth/metadata`; the
private key stays on the VM. Obtain the metadata verification certificate through
an independently trusted administrator channel and save it as `idp-metadata.crt`.
It is different from the IdP assertion certificate carried inside its metadata.

Production requires HTTPS metadata and verifies its signature with that trust
certificate. Refresh runs every six hours by default, earlier as expiry
approaches. A failed refresh retains the previous verified document only until
its `validUntil`; expired metadata blocks new SAML operations with 503. Startup
requires a successful fresh fetch. There is no disk fallback to stale metadata.

## Initialize and deploy

Run in the checkout, with permission to read `/etc/juena/juena.env` and use Docker:

```bash
./juena prod-config --quiet
docker compose -f docker-compose.yml --env-file /etc/juena/juena.env build juena-chatbot
./juena prod-repos-ingest jscatter
./juena prod-up
docker compose -f docker-compose.yml --env-file /etc/juena/juena.env --profile server ps -a
```

Ingestion uses the same image, `config/repositories.yaml`, and named clone cache.
It starts `rag-db` and `rag-init` automatically. Empty databases receive the pinned
RAG schema; existing ones are checked for required columns, indexes and embedding
width, without deleting or migrating data. Incompatible schemas stop deployment.
Chat schema setup remains the application's startup responsibility. For existing
chat databases, review schema changes before upgrading; startup is not a migration
system. Repeat ingestion with repository IDs, or omit IDs for the whole registry:

```bash
./juena prod-repos-ingest jscatter --dry-run --limit 10
./juena prod-repos-ingest jscatter
```

Dry runs clone and tokenize but do not embed/store. Normal runs call Blablador.
The first tokenizer use may download its files from Hugging Face. The running
chatbot's repository clones are re-pinned after ingestion. Schedule initial
ingestion before enabling sandbox workloads on the 8 GiB VM.

The bundled databases are new volumes. An old standalone RAG volume is not
automatically reused; export/restore a compatible corpus deliberately or re-ingest.
Never mount a PostgreSQL 16 data directory into PostgreSQL 18. The RAG PG18 volume
is mounted at `/var/lib/postgresql`, as required by the PG18 image layout.

## Install the worker

In the checkout, install the pinned worker environment as an account permitted
to write `.venv` (keep that environment readable/executable by `juena-sandbox`):

```bash
uv sync --frozen --no-dev
```

As `juena-sandbox`, build the image using its rootless Podman storage:

```bash
cd /opt/juena/juena-chatbot
podman build -t juena-sandbox:latest -f deploy/sandbox/Dockerfile .
systemctl --user enable --now podman.socket
mkdir -p ~/.config/systemd/user
cp deploy/juena-sandbox-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

As administrator, enable lingering (`loginctl enable-linger juena-sandbox`),
copy `deploy/sandbox-worker.env.example` to `/etc/juena/sandbox-worker.env`, and
make only this file readable by the worker (root:juena-sandbox, mode 0640).
Give the account directory traversal to `/etc/juena` (root:juena-sandbox, 0710),
while keeping the app environment and private-key directories inaccessible.
Set its loopback database credentials and the same concurrency, CPU, memory,
workspace and idle limits as the app. The user unit derives the Podman socket
from `%t`; do not copy another user's UID/socket path.

Configure real filesystem quotas before admitting generated-code workloads.
See [SANDBOX.md](SANDBOX.md) for ownership and isolation checks. Enable
`SANDBOX_ENABLED=true` in the app environment and re-run `./juena prod-up` to
create the worker's tables, then as `juena-sandbox`:

```bash
systemctl --user enable --now juena-sandbox-worker
systemctl --user status juena-sandbox-worker
journalctl --user -u juena-sandbox-worker -n 50
```

Do this during administrator acceptance, before allowing pilot users to execute.
Keep execution disabled if those checks fail. Restart policies recover processes;
they do not prove reboot recovery, quotas, or isolation on this VM.

## Capacity, persistence and upgrades

Start with tracing off, one sandbox slot at 1 CPU/1 GiB, chatbot at 3 GiB,
chat Postgres at 512 MiB, RAG at 768 MiB, pgvector at 1 GiB, and gateway at
128 MiB. Ingestion has a separate 2 GiB limit. These are limits, not measured
requirements; concurrent maxima plus the OS can exceed 8 GiB. Stop the worker
during ingestion and monitor real peak usage, OOM events and swapping.

Back up both databases with compatible `pg_dump` tooling, the artifact and audit
volumes, workspaces, configuration, the stable identity secret, and SAML keypair.
Repository caches can be rebuilt if their revisions remain available. Protect
backups containing credentials and test restoring into a separate Compose project.
VM snapshots alone do not prove database-consistent restoration.

Record the chatbot commit, core/RAG SHAs, image IDs/digests, environment revision,
and successful acceptance results for each deployment. Keep the reviewed image
tag immutable or archive/export it; do not rely on the mutable base-image tags for
rollback. `docker compose up -d --no-build` deploys a previously reviewed image.
Stop the host worker before upgrading its `.venv`, and restart it after the app's
schema checks. Back up before schema changes; image rollback cannot undo them.

`./juena prod-down` preserves all volumes. It does not stop the systemd worker;
stop that user service first for maintenance. Avoid `down --volumes` on retained
data. Configure certificate renewal with an nginx reload, disk/log monitoring,
metadata-refresh alerts, and a real VM reboot test.

SAML remains pending until IFFLogin registration and browser acceptance pass.
See [SAML-IFFLOGIN.md](../SAML-IFFLOGIN.md) and the local ignored checklist
`DOCS/DEV-SERVER-CHECKLIST.md`.

References: [PySAML2 configuration](https://pysaml2.readthedocs.io/en/latest/howto/config.html),
[Compose readiness ordering](https://docs.docker.com/compose/how-tos/startup-order/).
