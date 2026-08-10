# Sandbox implementation

The sandbox path has four parts:

1. JüNA stores an approved command as a `sandbox_jobs` row.
2. A host worker claims the row with `FOR UPDATE SKIP LOCKED` and runs one
   rootless Podman container.
3. The worker writes the terminal row: status, output, exit code, and the
   filenames it produced.
4. JüNA polls that row, validates the named files, and emits the existing status
   and artifact events to the UI.

There is no message broker. `sandbox_jobs` is the only channel, so there is no
second state to reconcile against it.

## Job state

`SandboxJobs` owns admission, claiming, and transitions.

```text
queued -> running -> completed | failed | timed_out
```

`busy` and `unavailable` are returned to the caller and never stored, so they
cannot occupy a slot.

Admission takes a transaction-scoped advisory lock, counts active rows, and
rejects at `SANDBOX_CONCURRENCY`. Partial unique indexes are the backstop for
one active job per opaque tenant and per workspace; their predicate is generated
from `ACTIVE_STATUSES` so a status change cannot leave them matching nothing.

There is no waiting queue. A `queued` row normally survives for one claim poll
(250 ms). Claims reject rows older than 15 seconds, and the worker fails stale
rows on that interval so abandoned work cannot execute later.

## One worker, detected without a heartbeat

The worker holds a two-int4 Postgres **session** advisory lock for its lifetime
on a connection outside the pool — a pooled connection is handed to other
callers after its transaction ends, and a session lock would travel with it. A
second worker fails to acquire it and exits.

JüNA detects worker presence by reading `pg_locks`:

```sql
SELECT 1 FROM pg_locks WHERE locktype = 'advisory'
  AND classid = %s AND objid = %s AND objsubid = 2 AND granted;
```

It never acquires the lock to test it. `pg_try_advisory_lock` followed by an
unlock would hold the lock for an instant, and a worker starting inside that
window would see it taken and exit permanently. `objsubid = 2` identifies the
two-integer form and is verified against a live Postgres in the integration
suite. Postgres releases the lock when the backend dies, so a killed worker is
detected immediately rather than after a heartbeat interval.

## Recovery

The worker never reattaches and never reruns. At startup it force-removes every
container labelled `io.juena.sandbox.managed` and fails every active row from
its previous life. During a run, the container is removed in a `finally` block.
Failure to remove a managed container stops the worker instead of releasing its
capacity.

## Files and isolation

`WorkspaceStore` is shared by JüNA and the worker. It validates owner markers,
absolute virtual paths, symlinks, file counts, file sizes, and workspace size.
Workspace guards make file operations return `workspace busy` while a job is
active instead of waiting behind it.

Three processes touch the workspace tree, so ownership is deliberate. The
sandbox container runs under `userns keep-id` mapped to the worker's own uid,
so files it writes are owned by the worker on the host instead of an unreachable
subuid. Directories are setgid `2770`, so files JüNA creates as root still
inherit the worker's group. `deploy/SANDBOX.md` has the details.

Generated containers keep the existing controls: rootless Podman, no network,
read-only root, all capabilities dropped, no-new-privileges, bounded
PIDs/CPU/memory/file size, no-exec temporary storage, read-only inputs, and a
single writable workspace. Both `/usr/bin/timeout` and the host worker enforce
the execution deadline. The application never receives a Podman socket.

The worker records changed artifact filenames only. JüNA reads those names from
the shared workspace and passes every file through the existing authenticated
artifact validator and event pipeline.

## Code map

| Responsibility | Module |
| --- | --- |
| Identity, admission, claiming, submit/poll | `src/juena/sandbox/jobs.py` |
| Statuses and artifact limits | `src/juena/sandbox/constants.py` |
| Shared files and owner markers | `src/juena/sandbox/workspace.py` |
| Podman controls and container lifecycle | `src/juena/sandbox/executor.py` |
| Host claim-loop worker | `src/juena/sandbox/worker.py` |
| Deep Agents and UI artifact bridge | `src/juena/sandbox/backend.py` |
| Host deployment | `deploy/SANDBOX.md` |

## Upgrading from the broker versions

`sandbox_jobs` is transient metadata with no release history, so there is no
migration: drop the table and let the application recreate it.

```sql
DROP TABLE IF EXISTS sandbox_jobs;
```

The current schema has no `expires_at` column and no `starting` status; an
existing table from the RabbitMQ version rejects every insert until it is
dropped. No other table is affected.
