# Sandbox worker

JüNA submits approved commands as durable `sandbox_jobs` rows. A host-side
worker claims those rows with `FOR UPDATE SKIP LOCKED` and runs each one under
rootless Podman. That table is the only channel between the two: there is no
message broker, so there is no second state to reconcile. The application
container has no Podman socket, and generated containers have no network.

This is a single-host design. The Postgres volume survives process and container
restarts, but loss of the host is loss of the service. It does not claim
exactly-once execution or high availability.

## Development

Install and initialize Podman once, then use the repository launcher:

```bash
brew install podman
podman machine init --memory 8192
./juena env
./juena up
```

The launcher:

1. starts the rootless Podman machine if necessary;
2. builds `deploy/sandbox/Dockerfile`;
3. starts JüNA and Postgres with Compose;
4. waits for JüNA to create `sandbox_jobs`;
5. starts `python -m juena.sandbox.worker` with two slots.

Use `./juena logs`, `./juena ps`, `./juena sandbox-stop`, and `./juena down`.
The worker PID and log are under `.sandbox/`.

`SANDBOX_CONCURRENCY` must be the same for JüNA and the worker: JüNA admits that
many active jobs before reporting `busy`, and the worker runs that many
containers. The Compose files and the launcher both default it to two in
development; production uses four in both places.

## Production host

1. Install rootless Podman and create the unprivileged `juena-sandbox` user.
2. Configure subordinate UID/GID ranges and enable lingering for that user.
3. Build `juena-sandbox:latest` as that user.
4. Enable its user Podman socket with `systemctl --user enable --now podman.socket`.
5. Put the shared workspace on a quota-capable filesystem:

   ```bash
   install -d -o juena-sandbox -g juena-sandbox -m 2770 \
     /var/lib/juena-sandbox/workspaces
   ```

6. Apply a hard 5 GB project/directory quota per workspace. The application
   size checks and per-file `fsize` limit do not replace a live filesystem quota.
7. Copy `deploy/sandbox-worker.env.example` to `/etc/juena/sandbox-worker.env`,
   set the database credentials, and make it readable only by the service
   account and administrator.
8. Install `deploy/juena-sandbox-worker.service`, then enable and start it only
   after the JüNA application has created the database schema.

## Shared workspace ownership

Three processes touch the workspace tree, so its ownership has to be deliberate:

- the **worker** owns it, running as `juena-sandbox`;
- the **sandbox container** runs under `userns keep-id` mapped to the worker's
  own uid, so files it writes are owned by the worker on the host rather than by
  an unreachable subuid;
- the **JüNA container** runs as root and reads the same tree through a bind
  mount at `/sandbox-workspaces`.

The setgid mode `2770` is what makes this work: files JüNA creates as root still
inherit the `juena-sandbox` group, so the worker and the sandbox container can
read and replace them. The systemd unit sets `UMask=0007` for the same reason.

Set `SANDBOX_WORKSPACE_HOST_ROOT` to the host path,
`SANDBOX_WORKSPACE_ROOT=/sandbox-workspaces` in JüNA, and the host path as
`SANDBOX_WORKSPACE_ROOT` in the worker. Never mount the Podman socket into JüNA.

## Capacity and behavior

- One active job per opaque tenant and per workspace, enforced by partial unique
  indexes.
- `SANDBOX_CONCURRENCY` jobs run at once. There is no waiting queue: the next
  submission returns `busy` without inserting a row.
- A submission returns `unavailable` without inserting a row when no worker
  holds the worker lock.
- Only one worker runs. It holds a Postgres session advisory lock for its
  lifetime; a second instance exits immediately, and Postgres releases the lock
  as soon as the process dies.
- Commands have the configured ten-minute ceiling both inside the container and
  in the worker.
- A worker restart force-removes every managed container and fails the jobs that
  were running. Nothing is ever reattached or rerun.
- Closing a browser can leave its current container running until the timeout,
  but cannot admit another job for that tenant.
- Worker failure does not stop chat startup; sandbox calls report
  `sandbox unavailable`.

## Production gate

Keep `SANDBOX_ENABLED=false` until these checks pass on the target host:

1. Four tenants complete concurrent jobs and a second job for one tenant is busy.
2. A fifth concurrent job returns `busy` immediately and inserts no row.
3. A second worker process exits without claiming anything.
4. An endless job times out while other tenants continue.
5. Killing the worker mid-job leaves no container and fails that job without
   rerunning it.
6. Stdout, stderr, PNG, and downloadable files reach the authenticated UI.
7. Recursive `/` searches are rejected and later workspace writes stay responsive.
8. JüNA starts and normal chat works with no worker running.
9. Old terminal job rows and idle workspaces are removed by the hourly pass.
10. The host has capacity for four configured sandbox CPU/memory limits plus
    JüNA, Postgres, and the operating system.
11. Generated code cannot reach networks, credentials, application data,
    container sockets, another tenant workspace, or arbitrary host paths.
