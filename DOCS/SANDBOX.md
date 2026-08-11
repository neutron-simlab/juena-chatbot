# JüNA sandbox: implemented architecture and production gates

This document is the maintained description of JüNA's scientific code-execution
sandbox. It replaces the former `SANDBOX-PLAN.md` and
`SANDBOX-IMPLEMENTATION.md`: the architecture is implemented, while the
remaining work is deployment validation on the target Linux host.

The operational installation procedure is kept separately in
[`deploy/SANDBOX.md`](../deploy/SANDBOX.md). This document explains the design,
security boundary, runtime lifecycle, artifact handling, tests, and the gates
that must pass before production enables the feature.

## Status

- The application, Postgres job channel, host worker, rootless-Podman executor,
  approval/resume flow, workspaces, artifact pipeline, audit records, and UI
  rendering are implemented.
- Development enables the sandbox through `docker-compose.dev.yml` and starts
  the host worker through `./juena up`.
- The base production configuration defaults to `SANDBOX_ENABLED=false`.
- Production must remain disabled until the target-host checklist in
  [Production enablement gates](#production-enablement-gates) passes.
- The design is single-host and intentionally does not claim high availability
  or exactly-once execution.

## Purpose and scope

The sandbox lets an authenticated user ask the software specialist to run
scientific Python or shell commands and receive validated plots and result files
in the chat. Model-authored commands must not gain access to JüNA credentials,
Postgres, container-engine sockets, application data, another user's files, or
the network.

Implemented use cases include:

- calculations with the fixed scientific Python environment;
- PNG plot generation;
- text, tabular, notebook, PDF, and NumPy result downloads;
- persistent scratch work across commands in the same chat workspace;
- read-only access to staged user inputs;
- user approval, editing, or rejection before an `execute` call;
- live status, artifact rendering, authenticated download, and chat reload.

The following are deliberately out of scope:

- runtime package installation or network access;
- arbitrary HTML, SVG, executables, or archives as delivered artifacts;
- GPU workloads;
- long-lived containers or notebook kernels;
- user-controlled images or container configuration;
- scientific correctness guarantees;
- multi-host scheduling or automatic worker failover.

## Security model

Human approval is not the security boundary. Repository text, uploaded data,
and prompts are untrusted, and the generated command may be wrong or hostile.
The container must remain contained even when the user approves such a command.

The main threats are:

1. prompt injection causing unintended execution;
2. cross-user or cross-thread file access;
3. access to host paths, application volumes, credentials, Postgres, or a
   container-engine socket;
4. network exfiltration;
5. CPU, memory, process, disk, output, or job-slot exhaustion;
6. stale or crashed work being executed again;
7. generated files being presented as delivered when the application rejected
   them.

The implemented controls are layered:

- authenticated runtime identity is converted into opaque tenant and workspace
  identifiers;
- Postgres admission and partial unique indexes isolate active tenants and
  workspaces;
- only the host worker receives the rootless Podman socket;
- generated containers have no network, a read-only root filesystem, dropped
  capabilities, `no-new-privileges`, and resource limits;
- `/inputs` is read-only and `/workspace` is the only writable bind mount;
- paths, owner markers, symlinks, file types, file contents, sizes, and image
  dimensions are validated;
- every output is either registered as an artifact or explicitly reported as
  not delivered.

The residual boundary is still a shared Linux kernel. Rootless Podman reduces
host privilege but is not a virtual machine. `SANDBOX_OCI_RUNTIME=runsc` can add
gVisor when it is installed and validated, but it is not assumed by the current
implementation.

## End-to-end architecture

```text
Authenticated user
    |
    v
Supervisor delegates calculations/files to software-specialist
    |
    v
FilesystemMiddleware exposes execute because the specialist's default backend
is RuntimePodmanSandboxBackend
    |
    v
HumanInTheLoopMiddleware pauses for approve / edit / reject
    |
    v
Approved command becomes a queued sandbox_jobs row in Postgres
    |
    v
Host worker claims the row with FOR UPDATE SKIP LOCKED
    |
    v
One ephemeral rootless-Podman container mounts:
  /workspace  read-write, persistent per user/thread
  /inputs     read-only, replaced atomically when uploads change
    |
    v
Worker removes the container, scans changed /workspace/outputs files,
stores bounded command output and accepted filenames in sandbox_jobs
    |
    v
Application polls the row, reads accepted files through the workspace guard,
validates them again, and copies them into the authenticated artifact store
    |
    v
SSE carries status and artifact metadata; PNGs may include bounded base64
    |
    v
Streamlit renders the final answer, plots, and authenticated downloads
```

Postgres `sandbox_jobs` is the only application-to-worker control channel.
There is no HTTP sandbox broker, RabbitMQ, AMQP, or engine socket in the JüNA
container.

## Agent and backend ownership

Execution belongs only to `software-specialist`. The supervisor keeps its
non-executing backend and cannot call `execute`; this limits the execution
surface to delegated software work.

The repository pins `deepagents==0.6.12`. Its `CompositeBackend.execute()`
delegates to the default backend, so the execution-capable sandbox must be the
software specialist's default rather than a path route. The implemented
composition is:

```text
CompositeBackend
  default: RuntimePodmanSandboxBackend
  routes:
    /inputs/ -> ReadOnlySandboxInputsBackend
    /repos/  -> read-only repository cache, when configured
    /skills/ -> read-only software-specialist skills, when authored
  artifacts_root: /workspace
```

`RuntimePodmanSandboxBackend` is a concrete `BaseSandbox`. It resolves trusted
`user_id`, `thread_id`, and `run_id` from the active LangGraph runtime for each
operation, then creates a tenant-bound `PodmanSandboxBackend`. A process-global
compiled graph therefore never captures one user's identity.

Unmatched filesystem paths reach the sandbox default, but that does not expose
host paths: the backend accepts only absolute virtual paths under `/workspace`
and read-only `/inputs`. Traversal, `~`, symlink escape, relative paths, and
recursive searches rooted at `/` are rejected.

## Identity and workspace isolation

The server derives two HMAC-SHA256 identifiers using
`SANDBOX_IDENTITY_SECRET`:

- `tenant_id = HMAC(user_id)` controls one active job per authenticated user;
- `workspace_id = HMAC(user_id + NUL + thread_id)` selects one chat workspace.

The raw user and thread identifiers are not used as directory names or worker
admission keys. The secret is required to contain at least 32 characters when
the sandbox is enabled.

Each workspace root contains:

```text
<workspace-id>/
  .owner             opaque tenant id
  workspace/         persistent read-write files
    outputs/         files intended for delivery
  inputs/            atomically replaced staged uploads
```

`WorkspaceStore` verifies the owner marker on every operation and rejects
symlinked workspace roots or owner files. Directories use mode `2770` and files
use `0660`. The setgid bit and the production worker's `UMask=0007` allow the
worker, its keep-id container, and the JüNA bind mount to share files without
making them world-readable.

Staged uploads are limited to 100 MB per replacement and mounted read-only in
the container. Replacing the input set uses temporary and old directories plus
atomic renames, so a command does not observe a partially staged set.

The writable workspace persists across commands and turns for the same
user/thread, but every command runs in a fresh container. Idle workspaces expire
after 24 hours by default. Active workspaces are excluded from cleanup, and
deleting a chat deletes its checkpoint, authenticated artifacts, and sandbox
workspace when the workspace guard is available.

## Approval and resume lifecycle

`HumanInTheLoopMiddleware` interrupts only the software specialist's `execute`
tool. The user can:

- **approve** the exact command;
- **edit** the command text and approve the edited command;
- **reject** it without creating a job.

The authenticated resume endpoint loads the pending LangGraph interrupt from
the owned thread. It rejects stale IDs, replayed decisions, malformed
interrupts, cross-user threads, non-`execute` actions, and empty edited
commands. The server constructs the edited action itself, keeping the tool name
and all policy-controlled execution settings unchanged.

The approval card can be restored after a browser reload from the checkpoint.
The resumed stream suppresses the interrupt ID just handled so it is not shown
again. A resume continues the existing turn and keeps its artifact budget;
starting a genuinely new message clears any abandoned in-memory artifact
window left by a cancelled stream.

Approval provides command visibility and compute consent. It does not protect
the host: inherited filesystem operations are not all approval-gated, and users
can approve unsafe code. Container and workspace controls remain authoritative.

`SandboxExecutionMiddleware` records the final, post-edit approved command as
`generated-command.sh` before execution and the model-visible tool output as
`execution-output.txt` afterwards. These are record artifacts, budgeted
separately from requested results.

## Postgres job control plane

The `sandbox_jobs` table stores transient execution metadata:

- UUID job ID;
- opaque tenant and workspace IDs;
- command, timeout, and maximum output bytes;
- status, output, exit code, and truncation flag;
- accepted artifact filenames;
- created, started, and finished timestamps.

The state machine is:

```text
queued -> running -> completed | failed | timed_out
```

`busy` and `unavailable` are immediate API results, not stored statuses:

- `unavailable` means no worker holds the worker-presence lock;
- `busy` means global capacity, the tenant, or the workspace cannot admit a new
  job.

There is deliberately no waiting queue. Admission obtains a transaction-scoped
global advisory lock, tries the workspace advisory lock, counts active rows,
and inserts only when capacity exists. Partial unique indexes over
`queued|running` rows enforce one active job per tenant and one per workspace as
the database backstop. Busy or unavailable requests insert no row and consume
no slot.

The worker claims the oldest fresh queued row using `FOR UPDATE SKIP LOCKED`.
Queued rows have a 15-second claim deadline; stale rows are never claimed and
are failed by the application or worker cleanup pass. The application polls at
250 ms until the row becomes terminal.

## Worker ownership, recovery, and cleanup

Only one worker process may operate. It holds a two-key Postgres session
advisory lock on a dedicated, non-pooled connection for its lifetime. The
application detects the worker by reading `pg_locks`; it never briefly acquires
the lock as a health probe. A second worker exits immediately.

At startup the worker:

1. refuses to run as root;
2. validates its concurrency, Podman socket, and fixed image;
3. force-removes every container labelled `io.juena.sandbox.managed`;
4. fails all queued or running rows left by the previous worker;
5. starts the configured execution-slot threads and one cleanup thread.

The worker never reattaches to a container and never reruns an interrupted job.
Every container is force-removed in `finally`. Failure to remove a managed
container is fatal to the slot and stops the worker rather than silently
releasing capacity.

Container creation and start are serialized because concurrent rootless-Podman
starts were unreliable on the development host. Execution remains concurrent
after start, so the lock does not reduce the configured number of running jobs.

Every 15 seconds the cleanup thread fails stale queued jobs. Hourly it removes
terminal rows older than 24 hours and idle workspaces older than the configured
TTL, excluding active workspaces.

## Container execution boundary

The worker creates one container per command with the following defaults:

| Control | Production default | Development launcher default |
| --- | ---: | ---: |
| Concurrent jobs | 4 | 2 |
| CPU quota per job | 2 vCPU | 1 vCPU |
| Memory per job | 4 GB | 2 GB |
| Process limit | 256 PIDs | 256 PIDs |
| `/tmp` tmpfs | 1 GB, `noexec` | 1 GB, `noexec` |
| Command timeout | 600 seconds | 600 seconds |
| Captured stdout/stderr | 100,000 bytes | 100,000 bytes |
| Workspace logical limit | 5 GB | 5 GB |
| Workspace idle TTL | 24 hours | 24 hours |

The generated container additionally has:

- rootless Podman and `userns=keep-id`;
- the worker UID rather than container root;
- `network=none`;
- a read-only root filesystem;
- all Linux capabilities dropped;
- `no-new-privileges`;
- a file-size ulimit equal to the workspace limit;
- only the read-write workspace and read-only input mounts;
- `HOME` and Matplotlib configuration under the disposable `/tmp` tmpfs;
- no application environment, repository mounts, credentials, or engine
  socket.

Both `/usr/bin/timeout` inside the container and the host worker enforce the
deadline. The worker allows a ten-second host grace period, then stops the
container and still removes it.

The fixed Python 3.11 image contains NumPy, SciPy, pandas, Matplotlib, seaborn,
SymPy, scikit-learn, Pillow, Grace, shell utilities, and the repository-backed
packages pinned in `src/juena/sandbox/repository-requirements.txt` (currently
`jscatter==1.9.0.1`). The image build smoke-tests JScatter and its headless
Grace plotting backend. `pip`, setuptools, wheel, and compilers are absent from
the final runtime image.

## Files and artifact delivery

### Producer contract

Commands may keep scratch files anywhere under `/workspace`. Only files the
user should receive belong under `/workspace/outputs`. `/tmp` is not persistent:
it is a fresh, `noexec` tmpfs for every command.

The software-specialist prompt forbids raw base64, data-image URIs, file paths,
`sandbox:` links, and Markdown images in its report. The application renders
validated artifacts itself.

### Worker scan

After a command finishes and its container has been removed, the worker scans
regular, non-symlinked files in `/workspace/outputs` whose modification time is
at or after the job start. It preserves older outputs without recollecting them,
so scripts and scratch work can be reused across commands.

The worker records only accepted filenames in `sandbox_jobs.artifact_names`.
Rejected names and reasons are appended to the existing job output, with at
most eight detailed rejections plus an additional-count summary. This covers
unsupported types, oversized files, the fifth PNG in one execution, and files
beyond the eight-file execution limit.

### Application validation and budgets

The application reads accepted filenames through the workspace guard and
validates their bytes again before copying them to the authenticated artifact
store.

| Boundary | Implemented limit |
| --- | ---: |
| PNGs accepted from one execution | 4 |
| Total files accepted from one execution | 8 |
| PNG size | 5 MB each |
| Downloadable file size | 10 MB each |
| PNG dimensions | 4096 px per edge and 16,000,000 pixels |
| Transfer to or from a workspace operation | 100 MB |
| Result images delivered in one turn | 8 |
| Result bytes delivered in one turn | 64 MB |
| Downloadable result count per turn | no count cap; byte cap applies |
| Execution-record artifacts per turn | 40 |
| Detailed queued undelivered notices | 8 plus omitted count |

The downloadable whitelist is:

```text
.csv .ipynb .json .log .md .npy .npz .pdf .py .sh .tsv .txt
```

PNG is the only inline image type. HTML, SVG, executables, and unapproved
archives are rejected. Text must be valid UTF-8 without NUL bytes; PDF, NPY, and
NPZ files must have the expected headers; PNGs are decoded and verified with
Pillow.

Turn accounting separates requested results from application bookkeeping. The
two execution records generated for every approved command do not consume the
eight-image or 64 MB result budget. If the 40-record budget is exhausted, the
command still runs; record persistence never blocks execution.

### No silent delivery failures

The delivery rule is: a relevant output is attached or explicitly reported as
not delivered.

There are several failure points, and each has a visible path:

1. **Worker scan rejection.** The worker appends the bounded filename/reason
   list to `sandbox_jobs.output` before saving the terminal row.
2. **Post-run collection failure.** Missing reads, a busy workspace, invalid
   content, or an application artifact-budget refusal is appended to the
   `execute` tool result seen by the specialist.
3. **Direct `write_file` registration failure.** The workspace write remains a
   success because the bytes exist. The backend audits the failed registration
   and queues an undelivered notice. A later approved execution reports it
   early; if there is no later execution, `SandboxExecutionMiddleware` appends
   it to the specialist's final report before the supervisor receives it.
4. **Abandoned turn.** A new user message clears a stale pending window from a
   cancelled or crashed stream; approval resume intentionally does not.

Shared `undelivered_note()` wording tells the model that the user cannot see the
file and forbids retry claims, paths, URLs, `sandbox:` links, or Markdown image
markup. Notice details are bounded to prevent a batch failure from flooding the
model context.

### Persistence, SSE, and UI

Accepted artifacts are copied eagerly out of the workspace into
`SANDBOX_ARTIFACT_ROOT`. Content and JSON metadata are stored under a directory
derived from the authenticated user ID. Retrieval requires that same user;
guessing an artifact ID does not cross the ownership check.

Artifact references are attached once to the supervisor's final `AIMessage`, so
the LangGraph checkpoint and chat history retain the metadata. The live event
queue sends an `artifact` SSE event for each file. Only validated PNGs include
base64 for immediate inline rendering; other files are fetched through the
authenticated artifact endpoint.

Streamlit merges live and stored artifact metadata by artifact ID, renders PNGs
beneath the final answer, provides download buttons, and avoids drawing the
same artifact twice during one rerun. Historical assistant text is sanitized:
dead local Markdown images are removed, `sandbox:` and `file:` links degrade to
their label, and valid HTTP(S) images remain untouched.

The artifact store is filesystem-backed rather than a database subsystem.
Artifact metadata and bytes remain until chat deletion; the workspace may
expire independently after its idle TTL because delivery uses the copied
artifact, not a live workspace path.

## Status and event streaming

The backend emits sandbox statuses through LangGraph's custom stream writer.
The server converts the graph stream into stable SSE payloads:

| Event | Purpose |
| --- | --- |
| `thread` | establishes the thread ID before other content |
| `approval_required` | command, limits, interrupt ID, and allowed decisions |
| `sandbox_status` | queued, running, completed, failed, timed out, busy, or unavailable |
| `artifact` | validated artifact metadata and optional inline PNG bytes |
| `token` / `message` | assistant output |
| `error` | user-visible stream failure |

The stream processor accepts current v2 dictionary events and legacy tuple
events. Specialist tokens and internal tool payloads are not shown as the user
answer; specialist work is represented as activity until its final report is
returned to the supervisor.

## Audit trail

`SANDBOX_AUDIT_FILE` is append-only JSONL. Implemented events include:

- `sandbox_approval`: user, thread, interrupt, original command, edited command,
  and decision;
- `sandbox_execution_approved`: authenticated identity, run, and the command
  that actually reached the execution middleware;
- `sandbox_execution`: job ID, command, terminal status, exit code, attached
  artifact IDs, and dropped filenames;
- `sandbox_execution_skipped`: a redundant base64/data-URI export that the
  application refused to run because output collection is automatic;
- `artifact_dropped`: a direct workspace output that was written but could not
  be registered, with its reason.

The command and execution output are also retained as bounded record artifacts
for the turn. Large scientific inputs and artifact contents are not duplicated
into the JSONL audit file.

Audit-file retention and rotation are deployment responsibilities; the
application does not currently rotate this file.

## Configuration and operations

### Application settings

| Variable | Meaning | Default |
| --- | --- | --- |
| `SANDBOX_ENABLED` | expose execution and initialize the application job pool | `false` |
| `SANDBOX_STARTER_PREVIEW` | show the plot starter without claiming execution works | development only |
| `SANDBOX_IDENTITY_SECRET` | HMAC tenant/workspace identity secret | required when enabled |
| `SANDBOX_WORKSPACE_ROOT` | workspace bind mount as seen by JüNA | repository data path |
| `SANDBOX_CONCURRENCY` | number of jobs the API admits | `4` |
| `SANDBOX_EXECUTION_TIMEOUT_SECONDS` | maximum requested command duration | `600` |
| `SANDBOX_MAX_OUTPUT_BYTES` | captured combined stdout/stderr | `100000` |
| `SANDBOX_ARTIFACT_ROOT` | copied artifact content and metadata | repository data path |
| `SANDBOX_AUDIT_FILE` | append-only JSONL audit file | repository data path |

### Worker settings

The worker reads `DATABASE_URL`, `SANDBOX_PODMAN_URI`, `SANDBOX_IMAGE`,
`SANDBOX_WORKSPACE_ROOT`, `SANDBOX_CONCURRENCY`, CPU, memory, PID, tmpfs,
workspace-limit, idle-TTL, and optional OCI-runtime settings. The application
and worker `SANDBOX_CONCURRENCY` values must match: one controls admission and
the other controls execution slots.

See [`deploy/sandbox-worker.env.example`](../deploy/sandbox-worker.env.example)
for the complete production environment and
[`deploy/juena-sandbox-worker.service`](../deploy/juena-sandbox-worker.service)
for the hardened systemd unit.

### Development

Install Podman once, then use:

```bash
brew install podman
podman machine init --memory 8192
./juena env
./juena up
```

`./juena up` starts the Podman machine when needed, builds
`deploy/sandbox/Dockerfile`, starts JüNA and Postgres with the development
Compose overlay, waits for `sandbox_jobs`, and starts the host worker. Use:

```bash
./juena ps
./juena logs
./juena sandbox-stop
./juena down
```

The worker PID, log, and default development workspace live under `.sandbox/`.

### Production

Production uses the base Compose file for JüNA and Postgres plus a separate
host-installed worker running as the unprivileged `juena-sandbox` account. The
shared workspace is bind-mounted into JüNA, but the Podman socket is never
mounted there.

Follow [`deploy/SANDBOX.md`](../deploy/SANDBOX.md) for host setup, directory
ownership, filesystem quota, environment, image build, systemd, and operational
commands.

## Verification

The repository has four validation levels.

### Unit and component tests

```bash
./juena test
```

This covers job behavior with fakes, backend routing, runtime identity,
workspaces, path and symlink confinement, executor container configuration,
worker recovery, approvals, middleware, artifact validation and budgets,
undelivered-file reporting, stream processing, authenticated artifact access,
and Streamlit rendering. At the time this document was consolidated, the exact
working tree passed `348` tests with `4` opt-in sandbox-pipeline tests skipped.

### Postgres integration

```bash
./juena test-integration
```

These tests use development Postgres and cover clean schema creation, exact
admission, no-row busy behavior, concurrent claiming, workspace/admission lock
interaction, stale rows, restart failure, the exclusive worker lock, and chat
ownership/persistence.

### Live Postgres-to-Podman pipeline

With a foreground worker and explicit test workspace:

```bash
JUENA_TEST_DATABASE_URL='postgresql://...' \
JUENA_TEST_WORKSPACE_ROOT='/absolute/shared/workspaces' \
JUENA_TEST_SANDBOX_CONCURRENCY=4 \
uv run --extra dev pytest -q tests/test_sandbox_pipeline_integration.py
```

The live suite exercises stdout/stderr and artifact delivery, full-slot busy
admission, tenant isolation during timeouts, and—when run separately with
`JUENA_TEST_WORKER_STOPPED=1`—the unavailable path.

### Static checks

```bash
git diff --check
uv run --extra dev ruff check src app tests
```

Render the merged development Compose configuration after deployment changes,
and use only the base Compose file for production validation.

## Production enablement gates

Implementation completeness is not production approval. Keep
`SANDBOX_ENABLED=false` on the target host until all of the following have
current evidence:

- [ ] Four configured tenants complete concurrent jobs on the production Linux
  host, and a second active job for one tenant is rejected as busy.
- [ ] The next job beyond global capacity returns busy immediately and inserts
  no row.
- [ ] A second worker exits without claiming work.
- [ ] An endless job times out while another tenant completes normally.
- [ ] Killing the worker mid-job removes its managed container, fails the row,
  and never reruns the command after restart.
- [ ] Network, DNS, Postgres, application volumes, secrets, Docker/Podman
  sockets, other tenant workspaces, and arbitrary host paths are unreachable
  from generated code.
- [ ] The production workspace filesystem enforces a hard 5 GB quota per
  workspace; application size checks alone are not treated as a quota.
- [ ] PNGs, downloadable results, approved-command records, bounded terminal
  output, and explicit rejection notices reach the authenticated UI and remain
  correct after chat reload.
- [ ] JüNA starts and ordinary chat remains usable with the worker stopped;
  sandbox calls report unavailable without creating rows.
- [ ] Hourly cleanup removes old terminal rows and idle workspaces while active
  workspaces remain untouched.
- [ ] The host has capacity for four configured CPU/memory limits plus JüNA,
  Postgres, and the operating system.
- [ ] A 30-session application load test with four simultaneous sandbox jobs
  shows bounded memory, stable Postgres pools, responsive chat, and immediate
  predictable busy responses.
- [ ] Audit logs are protected, rotated, retained, and monitored according to
  the deployment policy.
- [ ] The team explicitly accepts stock rootless-Podman shared-kernel risk or
  deploys and validates the configured gVisor runtime.

macOS Podman results are development evidence only. They do not prove Linux
kernel isolation, filesystem quota behavior, or production capacity.

## Code map

| Responsibility | Source |
| --- | --- |
| Statuses, types, and artifact limits | `src/juena/sandbox/constants.py` |
| Postgres admission, locks, claiming, polling, and cleanup | `src/juena/sandbox/jobs.py` |
| Database table and partial indexes | `src/juena/server/database/models.py` |
| Workspace ownership, staged inputs, paths, scans, and TTL | `src/juena/sandbox/workspace.py` |
| Podman container configuration and lifecycle | `src/juena/sandbox/executor.py` |
| Host claim-loop worker | `src/juena/sandbox/worker.py` |
| Tenant-bound Deep Agents backend and output collection | `src/juena/sandbox/backend.py` |
| Runtime identity and composite backend construction | `src/juena/sandbox/runtime.py` |
| Approval validation and resume command | `src/juena/sandbox/approvals.py` |
| Execution records and final undelivered notices | `src/juena/sandbox/middleware.py` |
| Authenticated artifact validation, storage, events, and audit | `src/juena/sandbox/artifacts.py` |
| SSE conversion | `src/juena/server/streaming/` |
| API start, resume, pending approval, download, and deletion | `src/juena/server/api/endpoints.py` |
| Streamlit approval and live artifact coordination | `app/chat_interface.py` |
| Artifact rendering and dead-link sanitization | `app/ui_components.py` |
| Development launcher | `juena` |
| Sandbox image | `deploy/sandbox/Dockerfile` |
| Production worker runbook | `deploy/SANDBOX.md` |

## Maintained design decisions

- Postgres is the sole worker channel; do not reintroduce an HTTP broker or a
  message queue beside `sandbox_jobs`.
- Containers are ephemeral per command; the workspace, not the container, is
  what persists across commands.
- Artifacts are copied eagerly into authenticated storage. A lazy `sandbox:`
  URL would break when the workspace expires and would make old chats depend on
  a live execution directory.
- `/workspace/outputs` is opt-in delivery. Scratch files elsewhere in
  `/workspace` remain reusable but are not attached.
- Downloadable result files have no per-turn count cap. The 64 MB turn budget,
  per-file validation, and eight-file execution scan are the intended controls.
- Execution records have a separate budget and may fail without blocking the
  approved command.
- Artifact failure reporting is deterministic application behavior, not a
  prompt-only promise and not a retry loop.
- Runtime package installation and generated-code network access remain
  forbidden. Add required scientific software to the fixed, reviewed image.
- A restart may reset the in-memory per-turn delivery budget, but it does not
  bypass per-file validation, workspace limits, or authenticated ownership.

## Legacy schema note

`sandbox_jobs` is transient metadata and this repository does not migrate old
experimental broker-era schemas. An existing development table with columns or
statuses from an earlier RabbitMQ/HTTP-broker implementation must be dropped so
normal schema creation can recreate the current table:

```sql
DROP TABLE IF EXISTS sandbox_jobs;
```

No user, chat, checkpoint, memory, or artifact table is affected by that
development-only reset.
