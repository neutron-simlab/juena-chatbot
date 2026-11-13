# Share an HTTPS gateway with another application

Use this topology when JüNA and another application, such as eLabFTW, share
one host and port 443. An independent `juena-edge` Compose project routes by
hostname over an external Docker network. Each application's databases stay
on its private network. For a dedicated JüNA server, use the standalone gateway
in [SERVER.md](SERVER.md); no shared project is needed.

All domains and certificate paths in `shared-proxy/` are examples. Keep actual
host mappings, migration commands, backup locations and acceptance results in
ignored `DOCS/`. The launcher reads installed configuration under
`/etc/juena-edge`, not these templates directly.

## Prepare a new shared gateway

1. Choose separate public HTTPS names for the applications. The templates use
   `juena.example.org` and `elab.example.org`. Install matching TLS material.
2. Review the existing application's routing and storage. If port 443 is already
   occupied, plan its migration before starting another listener. Preserve its
   configuration and a consistent database backup in a protected location.
3. Install the templates only into a new configuration directory:

   ```bash
   sudo install -d -m 0750 /etc/juena-edge
   sudo install -m 0644 deploy/shared-proxy/*.yml deploy/shared-proxy/*.conf /etc/juena-edge/
   ```

   Do not overwrite an existing gateway's configuration with example files.
   Edit the installed copies for the intended domains, TLS mounts, backend
   names/ports and passphrase handling before running them.
4. Create the external network if it does not already exist:

   ```bash
   sudo docker network create juena-edge
   ```

   Attach the existing application's web container to this network using its
   own managed configuration. The example expects alias `elabftw-web` on port
   8080; change it to the actual backend. Do not attach its database. JüNA's
   shared overlay adds alias `juena-app` automatically during deployment.
5. Validate and start the independent gateway using the appropriate selection
   below. It must already be running before `./juena prod up` can activate JüNA.

The example eLabFTW TLS mount is `/etc/elabftw/tls`, containing `fullchain.pem`
and `privkey.pem`. The JüNA mount is `/etc/juena/tls` with the same filenames.
The templates also show `ssl_password_file` for encrypted keys and a separate
eLabFTW password-file mount. Supply protected password files if required;
remove the matching directive and mount when the key is unencrypted. Keep key
and password bytes outside Git.

## Configuration selections

| Installed files | Behavior |
| --- | --- |
| `compose.yml` | Serve the existing application using `nginx.conf` |
| `compose.yml` + `compose.metadata.yml` | Also publish only JüNA's public SP metadata |
| `compose.yml` + `compose.juena.yml` | Route JüNA API/UI traffic using `nginx.with-juena.conf` |

For a new gateway serving only the existing application initially:

```bash
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml run --rm --no-deps edge nginx -t
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml up -d edge
```

Confirm the existing application remains reachable before adding JüNA.
Recreating the independent gateway briefly interrupts all applications it serves.

## Metadata before app startup

When the IdP requires a public metadata URL before registration, follow the
[offline SAML export workflow](SAML-IFFLOGIN.md#4-export-and-register-sp-metadata).
After verifying the XML signature, install the public document at
`/etc/juena/sp-metadata/sp.xml` and select the metadata-only overlay:

```bash
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml -f /etc/juena-edge/compose.metadata.yml run --rm --no-deps edge nginx -t
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml -f /etc/juena-edge/compose.metadata.yml up -d edge
```

It serves `/auth/metadata` while other JüNA routes return 503. It mounts only
public XML, not the SP private key. Confirm the public HTTPS response and its
signature before completing registration. Once app configuration and IdP trust
are ready, `prod up` switches to the full routing selection.

## Deploy and operate JüNA

Run from the checkout selected for deployment:

```bash
sudo ./juena prod config
sudo ./juena prod up
sudo ./juena prod ps
sudo ./juena prod health
sudo ./juena prod doctor
sudo ./juena prod logs
```

`prod up` builds this checkout, starts JüNA and its dependencies, and waits for
API/UI health. It validates the installed combined nginx configuration and TLS
keys before activating routing. App startup or nginx validation failure leaves
the current gateway configuration in place. Failed public HTTPS health checks
restore its previous Compose selection. This restores routing only; it does
not roll back application images or databases.

When the combined configuration is unchanged, Compose leaves the gateway
running. Switching from metadata-only routing recreates it. The running
container's Compose labels identify its current configuration selection; the
launcher refuses an unrecognized selection instead of guessing how to restore it.

`sudo ./juena prod down` stops the sandbox worker before JüNA's database and
keeps its data volumes. The shared gateway and other application keep running;
JüNA's hostname may return an upstream error while JüNA is stopped. Resume with
`sudo ./juena prod up`. A whole-host shutdown affects every hosted service.

## Source and installed settings

| Location | Purpose |
| --- | --- |
| `juena` | Launcher for app lifecycle, ingestion and sandbox operations |
| `docker-compose.shared.yml` | JüNA's external network attachment and gateway override |
| `deploy/shared-proxy/` | Generic gateway examples to customize during installation |
| `/etc/juena/juena.env` | Installed production settings, mode 0600 |
| `/etc/juena/saml/` | Persistent SP pair and independently confirmed IdP trust certificate |
| `/etc/juena/tls/` | JüNA HTTPS certificate chain, key and optional password file |
| `/etc/juena-edge/` | Installed independent gateway configuration |

The launcher uses its own checkout and does not clone or synchronize a second
source tree. Production settings come from `/etc/juena/juena.env`, not `.env`.
Use `sudoedit /etc/juena/juena.env` followed by `prod up` for app settings;
worker configuration changes require `prod sandbox-setup`. Keep database
passwords, the filesystem identity secret and registered SP keys stable.

Templates are not automatically copied onto the live gateway. Review changes
before installing them, preserving the existing application's configuration
and rollback material. This includes the `server_tokens` line and the
`add_header` lines in the JueNA server block of `nginx.with-juena.conf`: copy
them into the installed `/etc/juena-edge/nginx.with-juena.conf` by hand, and
keep `location` blocks free of their own `add_header`, which would drop them. Do not run competing lifecycle commands from multiple
checkouts against the same Compose project.

## Proxy selection

`PROD_PROXY_MODE=auto` selects shared mode when `/etc/juena-edge` exists and
standalone mode otherwise. Unreadable installed settings are an error. Shared
mode requires Compose 2.24.4+ for the overlay tags.

```bash
sudo env PROD_PROXY_MODE=shared PROD_EDGE_DIR=/etc/juena-edge ./juena prod config
```

`PROD_ENV_FILE` can select a different production environment. These are shell
options for the launcher, not settings read from the app's env file. Do not
select standalone mode while another gateway owns its HTTPS port.

## Renewal, diagnostics and recovery

After renewing certificates in mounted directories, validate and reload:

```bash
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml exec edge nginx -t
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml exec edge nginx -s reload
sudo docker compose -p juena-edge -f /etc/juena-edge/compose.yml logs --tail 100 edge
```

These `exec`/`logs` commands address the running edge container. For `up` or
`run`, include the overlay corresponding to the intended routing selection;
base-only recreation removes JüNA routing. Test both public hostnames after
changes and preserve the previous installed configuration for recovery.

Use [application troubleshooting](TROUBLESHOOTING.md) for SAML, RAG and worker
failures. Their fixes apply to both gateway modes. Record actual login/logout,
repository catalogue and plot-delivery results using the
[deployment record checklist](BUILD-RECORD.md#record-runtime-acceptance).
