# Sandbox worker

JüNA submits approved commands as durable `sandbox_jobs` rows. A host-side
worker claims those rows with `FOR UPDATE SKIP LOCKED` and runs each one under
rootless Podman. That table is the only channel between the two: there is no
message broker, so there is no second state to reconcile. The application
container has no Podman socket, and generated containers have no network.

This file covers worker operation, filesystem behavior and rollout checks.
Use [SERVER.md](SERVER.md#install-the-worker) for installation and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#a-plot-script-exists-but-no-png-was-generated)
for the failures encountered on the development server. These tracked guides
are usable from a fresh checkout without the ignored internal `DOCS/` notes.

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
5. starts `python -m juena_core.sandbox.worker` with one slot.

Use `./juena logs`, `./juena ps`, `./juena sandbox-stop`, and `./juena down`.
The worker PID and log are under `.sandbox/`.

`SANDBOX_CONCURRENCY` must be the same for JüNA and the worker: JüNA admits that
many active jobs before reporting `busy`, and the worker runs that many
containers. The Compose files, launcher and server template default to one.
Increase the slot count only after measuring capacity on the target host.

## Production host

Use `sudo ./juena prod sandbox-setup` from the deployed checkout. Setup
installs prerequisites and the dedicated account, builds/imports
the execution image, installs the protected worker environment and user service,
creates the optional queue, and enables execution only after a real PNG smoke
test. It runs the trusted worker in a rootless container with the application's
exact dependency image and the host network for loopback Postgres. Generated
commands run in separate containers with no network or container socket.
See [SERVER.md](SERVER.md#install-the-worker) for the steps, backups and failure
handling. No host virtualenv or administrator-home access is needed.
The same setup works with either the shared or standalone gateway.

`prod sandbox-test` repeats the queued plot check; `prod sandbox-logs` reads the
worker journal. `prod doctor` reports the application switch, queue table and
worker's database lock. `prod up` synchronizes worker dependencies with the
deployed app; `prod down` stops the worker before its database. Setting
`SANDBOX_ENABLED=false` and running `prod up` leaves the worker stopped.
Changes to the installed worker unit, generated-code image, database connection
or worker limits require `prod sandbox-setup` again. Ordinary `prod up` reuses
the installed unit and worker environment while updating its application image.

After setup, `prod doctor` should report all three sandbox flags as true. The
worker lock proves connection, not execution; run `prod sandbox-test` for the
queued PNG check, then test delivery of a plot through a new browser chat.
The Files panel and script writing remain available when execution is disabled.

The setup preserves the existing workspace filesystem and does not add hard
quotas. Application size checks and the per-file `fsize` limit are not a hard
per-workspace quota. Provision and verify that independently for broader rollout;
the smoke test does not establish it.

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
Production setup uses `crun` for the worker and execution containers, so the
runtime can enter the app-owned scratch directory as the configured job user.
The worker receives `SANDBOX_OCI_RUNTIME=crun`; `runc 1.2.2` fails its earlier
working-directory check on these restricted directories. Keep the `2770`/`0660`
workspace modes rather than broadening access to conversation files.

The parent `/var/lib/juena-sandbox` uses `0711` to allow remapped UIDs to
traverse it during rootless mount setup. This does not permit listing or writing
that parent, and does not change access to the restricted child directories.
Worker commands start from `/` to avoid inheriting the administrator's private
checkout. See the [permission failure explanations](TROUBLESHOOTING.md#rootless-worker-setup-failures-encountered).

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

## Broader rollout acceptance

Setup enables execution after its targeted smoke test. It does not establish
all of the following acceptance criteria on the target host:

1. A tenant completes a job and a second concurrent job for that tenant is busy.
2. With the one-slot configuration, another tenant also receives `busy` while
   that slot is occupied, without inserting a row.
3. A second worker process exits without claiming anything.
4. An endless job times out while other tenants continue.
5. Killing the worker mid-job leaves no container and fails that job without
   rerunning it.
6. Stdout, stderr, PNG, and downloadable files reach the authenticated UI.
7. Recursive `/` searches are rejected and later workspace writes stay responsive.
8. JüNA starts and normal chat works with no worker running.
9. Old terminal job rows and idle `/workspace/.scratch` contents are removed by
   the hourly pass; ordinary workspace files survive until conversation deletion.
10. The host has measured capacity for its configured sandbox slot plus JüNA,
    RAG, both databases, the gateway and the operating system.
11. Generated code cannot reach networks, credentials, application data,
    container sockets, another tenant workspace, or arbitrary host paths.

## Shared conversation filesystem rollout

Deploy the API, UI, and host worker from matching versions of `juena-core`.
Stop the old worker before upgrading: its cleanup loop deletes whole idle
workspaces. Start the updated API and worker together after the update.
The lockfile pins the same published core commit for the container and worker.
The production launcher imports the running app image into rootless Podman and
restarts the worker with that image. Native development still uses
`uv sync --frozen --no-dev` before restarting its host worker.

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
Native installation needs Node.js 24 and npm on PATH so the pinned core package
can build its frontend during `uv sync`; a sibling core checkout is unnecessary.
Keep `UI_PUBLIC_URL` and `JUENA_PUBLIC_URL` browser-reachable on the same cookie
host; their explicit origins configure credentialed CORS and browser-save origin
validation. Production uses the shared public origin. Never pass
`JUENA_API_INTERNAL_URL` or session tokens to the component.

The production gateway must route browser filesystem requests to FastAPI. Both
shipped nginx configurations forward `/auth`, `/chats`, `/threads`, `/artifacts`
and `/research` paths, `/health` and agent streaming/resume paths to FastAPI.
The `/threads` routing includes all filesystem requests. Other paths, including
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
