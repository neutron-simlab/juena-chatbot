# Linux development server

This deployment targets a full Linux VM on Proxmox, Docker Engine with Compose
2.20+ (2.24.4+ for the shared overlay), and a separate host worker using rootless
Podman. Only the chatbot repository is cloned. `uv.lock` fixes the published
core/RAG revisions; Docker builds core's Files frontend with Node.js 24.

Choose a standalone gateway for a dedicated JüNA host, or follow
[SHARED-PROXY.md](SHARED-PROXY.md) when sharing HTTPS with another application.
Production commands always build the checkout containing the invoked launcher.

Even a development server with real users runs with `ENVIRONMENT=production`
and IFFLogin authentication. Use `./juena prod ...` and the installed
`/etc/juena/juena.env`; `.env` and commands without `prod` are for local
workstations. See [troubleshooting](TROUBLESHOOTING.md) for known failures and
[deployment records](BUILD-RECORD.md) for what to record at each installation.
Keep site-specific hostnames, paths, capacity and acceptance evidence in ignored
`DOCS/`, outside the reusable runbooks and templates.

SAML validation, repository discovery and sandbox execution use the same
application code in both deployments. Only gateway selection and routing
differ. The PySAML2 callback fix requires a signed Response and verifies any
additional Assertion signature; that policy depends on the IdP, not nginx.

On a new dedicated server, the absence of `/etc/juena-edge` automatically
selects the standalone gateway. To select it explicitly, use the launcher
option on each production command, for example:

```bash
sudo env PROD_PROXY_MODE=standalone ./juena prod up
sudo env PROD_PROXY_MODE=standalone ./juena prod sandbox-setup
sudo env PROD_PROXY_MODE=standalone ./juena prod doctor
```

Use these examples on a dedicated server. Do not start a second gateway on
port 443 when an existing shared gateway owns it. `PROD_PROXY_MODE` is a launcher
environment option, not a setting loaded from `/etc/juena/juena.env`. Sandbox
activation passes the selected mode through its redeployment automatically.

For a different public hostname, update `JUENA_PUBLIC_URL`, `UI_PUBLIC_URL`
and the `SAML_SP_*` entity/endpoint URLs in the installed production environment,
install a matching TLS certificate, and coordinate the new SP metadata with
IFFLogin. No SAML code changes are needed for the new gateway or hostname.

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

Install Docker Engine, its Compose plugin and Git. The application build
provides Python, uv and Node.js. Use cgroup v2 with memory/CPU delegation.
The production sandbox setup below installs rootless Podman, its prerequisites,
and a dedicated unprivileged `juena-sandbox` account with subordinate UID/GID
ranges and lingering. It does not give that account Docker-group membership.

Choose one checkout for deployment, for example `/opt/juena/juena-chatbot`.
The launcher does not copy source to another directory. As administrator, create:

```bash
sudo install -d -m 0700 /etc/juena /etc/juena/saml /etc/juena/tls
sudo install -d -m 0700 /var/lib/juena-sandbox
sudo install -d -m 2770 /var/lib/juena-sandbox/workspaces
sudo install -m 0600 deploy/server.env.example /etc/juena/juena.env
```

The workspace is initially root-owned for the application. `sandbox-setup`
creates the worker account and grants its group access to the existing tree.
These commands initialize a new host. Do not overwrite an existing environment
with the example file or reset an installed worker's directory permissions.

For per-user credentials and upgrades, follow [personal API keys](PERSONAL-API-KEYS.md).

Edit `/etc/juena/juena.env`. Replace all `CHANGE_ME` values. Use distinct random
hex passwords (`openssl rand -hex 32`) for the databases and a separate stable
filesystem identity secret. These passwords appear in connection URLs, so hex
avoids URI escaping mistakes. Keep the identity secret and workspace path stable
across upgrades. Never use the development bypass on this server.

The `gateway` container supplies HTTPS for JueNA. In the server environment,
`JUENA_TLS_DIR=/etc/juena/tls` selects the host certificate directory. Compose
mounts it read-only into nginx:

```yaml
- ${JUENA_TLS_DIR:-./.tls}:/etc/nginx/tls:ro
```

Place these PEM files in that directory:

| Host file | Purpose / format |
| --- | --- |
| `/etc/juena/tls/fullchain.pem` | Server certificate first, followed by intermediate CA certificates in issuer order |
| `/etc/juena/tls/privkey.pem` | Matching private key; protect it with mode 0600 and keep it readable by nginx's master process |

The certificate must cover the actual development hostname. The configured
container filenames are fixed in `deploy/nginx.conf`:

```nginx
listen 443 ssl;
ssl_certificate /etc/nginx/tls/fullchain.pem;
ssl_certificate_key /etc/nginx/tls/privkey.pem;
```

The `server` Compose profile enables this gateway. JueNA uses nginx's TLS
configuration directly; eLabFTW's `DISABLE_HTTPS`, `TLS_CERT_PATH` and
`TLS_KEY_PATH` variables do not configure JueNA. Backend UI/API traffic remains
HTTP on the private Docker network. Keep HTTPS keys out of the repository and
separate from the SAML signing keypair below.

The default configuration expects a key loadable without an interactive prompt.
For a passphrase-protected key, place its passphrase on one line in a protected
`/etc/juena/tls/tls-password` file (mode 0600), and enable the documented line
in `deploy/nginx.conf`:

```nginx
ssl_password_file /etc/nginx/tls/tls-password;
```

That password file is included in the existing read-only directory mount; do
not put its contents in the Compose environment, Git or logs. See nginx's
[certificate chain format](https://nginx.org/en/docs/http/ngx_http_ssl_module.html#ssl_certificate)
and [encrypted key support](https://nginx.org/en/docs/http/ngx_http_ssl_module.html#ssl_password_file).

With the real certificate, key and server environment installed, validate the
gateway without publishing ports or starting application dependencies:

```bash
sudo docker compose -f docker-compose.yml --env-file /etc/juena/juena.env --profile server run --rm --no-deps gateway nginx -t
```

This checks nginx configuration/key loading, not browser trust or SAML acceptance.
After renewing files in the mounted directory, test and reload the running nginx
with the same Compose selection (`exec gateway nginx -t`, then
`exec gateway nginx -s reload`). If TLS must terminate at an institutional proxy,
agree that topology first and adapt the gateway and forwarded-header trust;
this template does not configure that alternative.

## Register SAML authentication

Follow [SAML-IFFLOGIN.md](SAML-IFFLOGIN.md) to agree the public URLs and access
scope, install a persistent SP keypair and independently confirmed IdP trust
certificate, export signed metadata and complete registration. The guide also
covers metadata publication before app startup and browser acceptance.

Keep the SP key stable across rebuilds and coordinate replacement with the
IdP. HTTPS credentials and the SAML signing pair have separate roles. The app
requires fresh, signed IdP metadata at startup and refreshes it periodically;
expired metadata blocks SAML operations. Do not copy a fingerprint, certificate
or registration status from another installation's history.

## Initialize and deploy

Run in the checkout, with permission to read `/etc/juena/juena.env` and use Docker:

```bash
sudo ./juena prod config
sudo ./juena prod repos-ingest jscatter
sudo ./juena prod up
sudo ./juena prod ps -a
sudo ./juena prod health
sudo ./juena prod doctor
```

`prod config` validates quietly by default; `--show` prints resolved settings,
including secrets. `prod repos-ingest` and `prod up` build the current source as
needed. Use `prod build` to build without starting services. `prod logs` follows
app logs, `prod logs juena-rag` follows retrieval logs, and `prod health` checks
public API/UI HTTPS. The earlier `prod-*` command names remain aliases.

When `/etc/juena-edge` exists, the launcher merges `docker-compose.shared.yml`,
keeps the standalone gateway disabled, waits for the app's health checks and
then activates the installed shared gateway configuration. `prod down` stops
the JueNA project and retains its volumes; eLabFTW and the shared gateway stay
running. See [SHARED-PROXY.md](SHARED-PROXY.md) for mode selection and recovery.

Ingestion uses the same image, `config/repositories.yaml`, and named clone cache.
It starts `rag-db` and `rag-init` automatically. Empty databases receive the pinned
RAG schema; existing ones are checked for required columns, indexes and embedding
width, without deleting or migrating data. Incompatible schemas stop deployment.
Chat schema setup remains the application's startup responsibility. For existing
chat databases, review schema changes before upgrading; startup is not a migration
system. Repeat ingestion with repository IDs, or omit IDs for the whole registry:

```bash
sudo ./juena prod repos-ingest jscatter --dry-run --limit 10
sudo ./juena prod repos-ingest datreat mdanse sasview sasmodels sasdata silx vitess
sudo ./juena prod doctor
```

The seven-ID example fills the remaining registry after the initial jscatter
ingest. `sudo ./juena prod repos-ingest` processes all eight configured entries
sequentially; unchanged revisions are skipped. Use `--force` after changing
chunking or file-selection settings without a new repository commit. Avoid
overlapping ingestion runs. Startup and `prod up` never index new content.
Re-ingest every two weeks, at least monthly, and after relevant upstream releases.

Dry runs clone and tokenize but do not embed/store. Normal runs send embedding
batches to Blablador, using `alias-qwen3-8b-embeddings` by default. Hugging Face
supplies tokenizer files for local token counting, not the embedding model or
an inference service. An anonymous-download warning needs no action if the
download succeeds. The running chatbot's repository clones are re-pinned after
ingestion. See [repository maintenance](../HOW-TO-ADD-REPO.md) for details.

In `prod doctor`, compare every configured ID with `indexed` and its nonzero
chunk count. `cloned` only proves files are present. `needs_ingestion=false`
means at least one repository has chunks; it does **not** mean every configured
repository is indexed. A healthy RAG service can still have an incomplete corpus.

The bundled databases are new volumes. An old standalone RAG volume is not
automatically reused; export/restore a compatible corpus deliberately or re-ingest.
Never mount a PostgreSQL 16 data directory into PostgreSQL 18. The RAG PG18 volume
is mounted at `/var/lib/postgresql`, as required by the PG18 image layout.

## Install the worker

With the application running, install and activate the worker from this checkout:

```bash
sudo ./juena prod sandbox-setup
sudo ./juena prod doctor
```

This command installs Podman, crun and rootless prerequisites on an apt-based Linux
host, creates `juena-sandbox` with home `/var/lib/juena-sandbox/home`, checks its
subordinate IDs, enables lingering and the user socket, and grants group access
to the existing `/var/lib/juena-sandbox/workspaces` tree. It preserves workspace
contents and the filesystem identity secret. Other workspace roots or existing
accounts with a different home require their own installation review; the
installer refuses to change permissions on arbitrary directories.
Keep `SANDBOX_ENABLED=false` for a first installation until this command enables
it. Merely installing Podman or running `prod up` does not install the worker.

The worker runs as that unprivileged UID inside a rootless container using the
running application's exact image. This provides the pinned core dependencies
without a host `.venv`, Node.js, or a second source checkout. The execution image
is built from this checkout with Docker and imported into that account's Podman
storage. Only the trusted worker receives the rootless socket and its loopback
database credentials; the application and generated-code containers do not.
Commands for the worker account start in `/`, so deployment also works from a
checkout under a private administrator home without widening its permissions.
The root-owned `/var/lib/juena-sandbox` parent uses mode `0711`: remapped
container users need traversal permission during mount setup. This grants
other users neither directory listing nor write access. The worker home and
workspace tree retain their restricted permissions.
Both the worker container and execution jobs explicitly use `crun`. The
`runc 1.2.2` runtime attempts to create/check the job's working directory
before switching to its UID/GID, which fails for app-owned `2770` workspaces.
`crun` retries access after switching to the job's user. Setup installs it even
when Podman is already present, and writes `SANDBOX_OCI_RUNTIME=crun` into the
worker environment. No global Docker or Podman runtime configuration is edited.
The dedicated account's Podman API is restarted before the worker connects, so
it discovers the installed runtime even after an earlier failed setup attempt.
Before starting the worker, the launcher tests a non-root container with the
workspace mounted read-only and no credentials or Podman socket, reporting
runtime errors directly. `prod sandbox-logs` prints complete journal lines
without an interactive pager.

The launcher derives `/etc/juena/sandbox-worker.env` from the live database URL
and production limits, writes it root:juena-sandbox 0640, and installs the user
unit. It creates only the optional `sandbox_jobs` table when absent. Before
enabling execution, it submits a disposable plot through the real queue and
requires successful non-root execution, blocked outbound networking, absence
of app credentials/socket in the execution container, valid PNG bytes, and
the worker's artifact report. Failed tests leave the application setting
unchanged. Success backs up `/etc/juena/juena.env`, changes only
`SANDBOX_ENABLED`, redeploys through `prod up`, and tests again. An activation
failure restores the previous environment; installed packages/accounts remain.

For operations:

```bash
sudo ./juena prod sandbox-test       # disposable queued plotting test
sudo ./juena prod sandbox-logs       # worker journal; append -f to follow
sudo ./juena prod up                 # stop worker, deploy, sync its image, restart
```

The diagnostic should report `enabled`, `jobs_table_exists`, and
`worker_connected` as `true`. The connection check observes a database lock;
`sandbox-test` is the separate proof that a queued command creates a PNG.
After deployment, refresh the browser and try a plot in a new conversation,
approve its execution, and confirm the resulting image is delivered. A script
saved in the Files panel does not prove it was executed.

The first build downloads scientific packages and can take several minutes.
The setup does not provision hard filesystem quotas or prove reboot recovery
and every isolation scenario. Existing application size checks and per-file
limits remain; broader rollout checks are in [SANDBOX.md](SANDBOX.md). Keep
those limits distinct from a hard per-workspace filesystem quota.

## Routine updates and shutdown

For source or ordinary application-setting changes:

```bash
sudoedit /etc/juena/juena.env          # only if settings need changing
sudo ./juena prod config
sudo ./juena prod up
sudo ./juena prod health
sudo ./juena prod doctor
sudo ./juena prod sandbox-test        # when sandbox execution is enabled
```

`prod up` rebuilds from this checkout and recreates services as needed. It stops
the worker first, imports the running app's image into rootless Podman, then
starts the worker if execution is enabled. Wait for active jobs and ingestion
to finish before maintenance. Changes to the worker unit, sandbox image, worker
limits or database connection require `prod sandbox-setup` again: `prod up`
does not rewrite the installed unit or worker environment.

For a planned stop, use `sudo ./juena prod down`. It stops and disables the
worker before stopping JueNA's Compose services and preserves named volumes.
In shared mode eLabFTW and the independent gateway continue running; in
standalone mode JueNA's gateway stops too. Use `sudo ./juena prod up` to resume.
A whole-VM shutdown also interrupts eLabFTW and every other service on that VM;
coordinate that separately. Avoid `down --volumes` on retained data.

## Capacity, persistence and rollback

Measure the target host's memory with `free -h` and account for every service
sharing it. Available memory changes with workloads and caches. Increasing VM
memory does not raise container limits automatically. Keep actual capacity and
measurement results in the private deployment record.

Start with tracing off, one sandbox slot at 1 CPU/1 GiB, chatbot at 3 GiB,
chat Postgres at 512 MiB, RAG at 768 MiB, pgvector at 1 GiB, and gateway at
128 MiB. Ingestion has a separate 2 GiB limit. These are limits, not measured
requirements. Keep the one-slot setting until workload measurements justify a
change, schedule ingestion away from heavy interactive execution, and monitor
peak usage, OOM events and swapping. Tracing remains off in the server template.

Back up both databases with compatible `pg_dump` tooling, the artifact and audit
volumes, workspaces, configuration, the stable identity secret, and SAML keypair.
Repository caches can be rebuilt if their revisions remain available. Protect
backups containing credentials and test restoring into a separate Compose project.
VM snapshots alone do not prove database-consistent restoration.

Record the chatbot commit, core/RAG SHAs, image IDs/digests, environment revision,
and successful acceptance results for each deployment. Keep the reviewed image
tag immutable or archive/export it; do not rely on the mutable base-image tags for
rollback. The launcher rebuilds on `prod up`; it has no general image-only
rollback command. Restore a reviewed source revision and compatible settings
before redeployment, or prepare an image-only recovery using the same Compose
files, project, environment and proxy mode. Keep the worker stopped until its
image matches the recovered app. Back up before schema changes; changing an
image cannot undo them. Shared-gateway routing restoration also does not roll
back application images or data.

Configure certificate renewal with an nginx reload, disk/log monitoring,
metadata-refresh alerts, and a real VM reboot test. Use the tracked
[SAML acceptance checklist](SAML-IFFLOGIN.md#acceptance-checklist) and
[sandbox rollout checks](SANDBOX.md#broader-rollout-acceptance).

References: [PySAML2 configuration](https://pysaml2.readthedocs.io/en/latest/howto/config.html),
[Compose readiness ordering](https://docs.docker.com/compose/how-tos/startup-order/).
