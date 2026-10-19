# Sandbox worker

JüNA submits approved commands as durable `sandbox_jobs` rows. A host-side
worker claims those rows with `FOR UPDATE SKIP LOCKED` and runs each one under
rootless Podman. That table is the only channel between the two: there is no
message broker, so there is no second state to reconcile. The application
container has no Podman socket, and generated containers have no network.

See [`DOCS/SANDBOX.md`](../DOCS/SANDBOX.md) for the implemented architecture,
threat model, artifact lifecycle, verification layers, and production gates.
This file is the host installation and operations runbook.

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
5. starts `python -m juena_core.sandbox.worker` with two slots.

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
9. Old terminal job rows and idle `/workspace/.scratch` contents are removed by
   the hourly pass; ordinary workspace files survive until conversation deletion.
10. The host has capacity for four configured sandbox CPU/memory limits plus
    JüNA, Postgres, and the operating system.
11. Generated code cannot reach networks, credentials, application data,
    container sockets, another tenant workspace, or arbitrary host paths.

## Shared conversation filesystem rollout

Deploy the API, UI, and host worker from matching versions of `juena-core`.
Stop the old worker before upgrading: its cleanup loop deletes whole idle
workspaces. Start the updated API and worker together after the update.
The launcher installs its sibling core package; for native development run
`uv sync --extra dev --reinstall-package juena-core` in `juena-chatbot`.

Keep the existing workspace root and identity secret unchanged. A stable secret of
at least 32 characters is now required even when sandbox execution is disabled.
If an older deployment used an empty secret, stop and migrate its workspace
identities before setting a new secret; do not start it against newly keyed empty
workspaces. Back up the old root and secret before any such migration. Existing disk
files stay at their paths. Legacy checkpoint uploads are copied on first access
into `/workspace/uploads`; collisions preserve both contents using a suffix.
A persisted alias map makes migration repeatable and keeps old `/inputs/uploads`
paths readable. Missing alias targets are omitted without resurrecting deleted
uploads from old checkpoints. Subsequent edits update only the workspace copy.

The Files panel beside chat and specialist tools use the same disk files, even
with `SANDBOX_ENABLED=false`. The existing root, quota, and TTL settings still
apply. TTL now affects only `.scratch`; the API also runs hourly cleanup when
no execution worker is present. Back up the persistent workspace alongside the
database and artifact store. Changing the identity secret changes workspace
identifiers, so it requires a separate storage migration.

The UI uses a 60/40 chat/files layout with a React folder browser and Ace editor.
Docker builds core's frontend in a Node 24 stage; the running image needs no Node.
For native installation, first run `npm ci && npm run build` in
`juena-core/frontend/filesystem`, then reinstall core in chatbot's environment.
Keep `UI_PUBLIC_URL` and `JUENA_PUBLIC_URL` browser-reachable on the same cookie
host; their explicit origins configure credentialed CORS and browser-save origin
validation. Production uses the shared public origin. Never pass
`JUENA_API_INTERNAL_URL` or session tokens to the component.

The production gateway must route browser filesystem requests to FastAPI. With
the VM plan's unprefixed URLs, Caddy must forward `/auth/*`, `/chats/*`,
`/threads/*/filesystem`, and `/threads/*/filesystem/*`; other paths, including
`/_stcore/*`, go to Streamlit. Forwarding only `/auth/*` is insufficient. On
OpenShift (or a gateway using the same convention), route `/api/*` to FastAPI
and set `JUENA_PUBLIC_URL=https://<host>/api`, with `UI_PUBLIC_URL=https://<host>`.
Verify authenticated list, preview, download and Save through the public hostname.
Use explicit UI origins for credentialed requests; origin checks compare origins,
not the `/api` path prefix.

Browsing uses independent HTTP requests during response streaming. During a
sandbox command, cached views remain visible and uncached requests show that
files are busy. The visible folder and clean preview refresh at turn completion
or an approval pause. Drafts remain separate per session and conversation; no
continuous filesystem polling or data migration is needed for the React update.

Commands start in `/workspace/.scratch`. Use absolute `/workspace/...` paths
for durable scripts, uploads, and outputs. An execution's nested outputs retain
their relative paths. Chat artifacts remain immutable snapshots; the browser
shows current contents. File preview and Save never execute code.

API and worker coordinate through OS file locks in the shared storage and the
existing execution queue. Use a single host or storage with equivalent lock
semantics. Active operations and pending approvals protect scratch from expiry;
user saves return a conflict during either. Concurrent reads share a lock; short writes wait briefly without blocking the
API event loop. The execution gate alone causes immediate busy responses to
file reads. Cancellation and normal completion reconcile the pause marker with
the checkpoint. A process killed before reconciliation can still leave a pending
marker; resume the conversation or complete an allowed save to reconcile it.
Deletion tombstones prevent late workers and stale bookmarks from recreating a
deleted workspace. Lock files and tombstones intentionally remain outside the
workspace: unlinking a lock with waiters splits coordination across inodes, and
removing a tombstone would allow retired identities to be reused.

Editing supports UTF-8 text up to 1 MiB, with revision checks and atomic saves.
Larger text has a bounded preview. PNG previews are bounded to 5 MiB; downloads
are bounded to 100 MiB. HTML and notebooks display source. Temporary files,
pasted inputs, and specialist notes are read-only in the browser. Existing
upload type, count, and size limits continue to apply.

Quota accounting is cached outside the container mount and invalidated around
sandbox commands and scratch cleanup. After an administrator changes workspace
files directly or restores only part of a workspace, remove that workspace's
`.usage` file before restarting services so the next write recounts disk usage.
Draft text stays in the browser's session-scoped cache rather than crossing the
Streamlit bridge on each pause in typing. Permission failures preserve it; session
expiry clears it and explicitly tells the user that unsaved edits were discarded.
