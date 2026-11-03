# JueNA and eLabFTW on the shared gateway

The iff1122 VM has one independent Docker Compose project, `juena-edge`, serving
both HTTPS names on port 443. It forwards `iff1122.iff.kfa-juelich.de` to JueNA
and `jcns-elabs.fz-juelich.de` to eLabFTW over the external `juena-edge` network.
The app databases remain on their private networks.
For a dedicated server, use the standalone gateway in [SERVER.md](SERVER.md).
Authentication, repository indexing and sandbox execution do not depend on this
shared project. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) covers their common fixes.

## Normal operation

Use the existing launcher from the checkout you intend to build:

```bash
cd /home/administrator/juena-chatbot
sudo ./juena prod config
sudo ./juena prod up
sudo ./juena prod ps
sudo ./juena prod health
sudo ./juena prod logs
```

`prod up` builds from this checkout, starts JueNA and its dependencies, and waits
for API/UI health. It validates the installed combined nginx configuration and
TLS keys before activating routing. An app startup or nginx validation failure
leaves the current gateway configuration in place. Failed public HTTPS health
checks restore the previous gateway Compose selection. This restores proxy
routing only; it does not roll back application images or databases.

When already using the combined configuration, Compose leaves an unchanged
gateway running. The first switch from metadata-only routing briefly recreates
it, affecting both HTTPS services. The running container's Compose labels are
the authority for its current configuration selection.

To stop JueNA and retain its data:

```bash
sudo ./juena prod down
```

This stops the sandbox worker before JueNA's database and leaves the shared
gateway and eLabFTW running. JueNA's hostname returns an
upstream error while its app is stopped. Use `sudo ./juena prod up` to start it
again. Do not use `--volumes` when stopping a deployment whose data you need.

Additional commands:

```bash
sudo ./juena prod build                         # build only
sudo ./juena prod doctor                        # safe RAG/worker diagnostics
sudo ./juena prod logs juena-rag                # retrieval logs
sudo ./juena prod repos-ingest jscatter         # explicit ingestion
sudo ./juena prod repos-ingest jscatter --dry-run --limit 10
```

Ingestion builds the same image and initializes the retrieval schema if empty.
It uses Blablador's embedding API; normal startup does not re-ingest. On this VM,
schedule ingestion away from heavy interactive/worker workloads.

Use [the RAG diagnostic guide](TROUBLESHOOTING.md#rag-is-healthy-but-repositories-or-search-tools-are-missing)
to distinguish configured, cloned and indexed repositories. The wheel-path fix
and retained repository tools apply equally to standalone deployments.

## Source, configuration and secrets

| Location | Purpose |
| --- | --- |
| `juena` | Bash launcher for development and production commands |
| `docker-compose.yml` | App, databases, retrieval and optional standalone gateway |
| `docker-compose.shared.yml` | App's shared network attachment; disables standalone gateway |
| `deploy/shared-proxy/` | Public gateway configuration templates for this VM |
| `/etc/juena/juena.env` | Installed production settings and credentials, mode 0600 |
| `/etc/juena/saml/` | Registered SP keypair and confirmed IFFLogin trust certificate |
| `/etc/juena/tls/` | HTTPS certificate chain, private key and optional password file |
| `/etc/juena-edge/` | Installed independent gateway configuration |

The launcher always uses the checkout it belongs to, even when invoked through
a symlink or from another directory. It does not clone, copy or synchronize a
second source tree. The initial installation created `/opt/juena/juena-chatbot`;
retain it as an installation snapshot. Subsequent commands from the working
checkout use the same `juena-chatbot` project and named data volumes. Avoid
running lifecycle commands from both checkouts concurrently.

The installed `/etc/juena/juena.env` is authoritative. Editing the working
checkout's `.env` does not change production; use `sudoedit /etc/juena/juena.env`
and then `sudo ./juena prod up`. Keep existing database passwords, filesystem
identity secret and registered SAML keys stable across updates.

No deployment command depends on `.sandbox` or a separate Python installer.
Old migration artifacts are retained for recovery, outside the ongoing workflow.

## Proxy selection

`PROD_PROXY_MODE=auto` is the default: the presence of `/etc/juena-edge` selects
shared mode; otherwise the launcher enables JueNA's standalone `server` profile.
An unreadable installed environment/configuration is an error, not a fallback.
Shared mode requires Compose 2.24.4+ for the overlay tags (this VM has 2.29.7).

For a different installation, explicit shell settings are available:

```bash
sudo env PROD_PROXY_MODE=shared PROD_EDGE_DIR=/etc/juena-edge ./juena prod config
```

`PROD_ENV_FILE` can select a different production environment. These are launcher
options supplied in the shell, not settings read from the application's env file.
Do not select standalone mode on this VM while the shared gateway owns port 443.

## Installed gateway and certificate renewal

The full gateway uses `/etc/juena-edge/compose.yml` followed by
`compose.juena.yml`; the latter selects `nginx.with-juena.conf` and mounts JueNA's
TLS directory. Base-only recreation selects eLabFTW-only routing. The installed
metadata-only files remain available as a fallback before app activation.

Public templates under `deploy/shared-proxy/` preserve these same configurations
for review. They contain the VM's hostnames and paths, not certificate/key bytes.
They are not read directly by the live container or automatically copied over
installed administrator configuration. Review and install intended gateway
changes under `/etc/juena-edge`, then validate them before reload/recreation.

After renewing certificates in their mounted directories:

```bash
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml exec edge nginx -t
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml exec edge nginx -s reload
```

Gateway logs are independent of the app:

```bash
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml logs --tail 100 edge
```

The original eLabFTW migration backup is
`/var/backups/juena-edge/20261103T121100Z-7163`. Its configuration and consistent
MySQL dump remain protected. eLabFTW still uses `/etc/elabftw.yml` and `elabctl`;
changing JueNA source or stopping JueNA does not edit that configuration.

## Current acceptance

The initial 3 November 2026 installation established public API/UI health,
metadata and the login redirect while preserving eLabFTW. Later work fixed the
ACS signature/PySAML2 issue, registry paths and worker startup. At 18:03 UTC, the
user's `prod doctor` output reported all sandbox flags true, all eight repository
IDs configured/cloned, and jscatter indexed. Ingestion of the remaining seven
was subsequently started; its completion has not yet been recorded.

See [the deployment record](BUILD-RECORD.md#2026-11-03--auth-retrieval-and-sandbox-recovery)
for evidence and remaining browser/SLO/plot checks. A shared-gateway health check
does not establish those application behaviors.
