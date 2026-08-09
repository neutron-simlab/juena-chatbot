# JüNA Sandbox Implementation

This document records what was implemented from
[SANDBOX-PLAN.md](SANDBOX-PLAN.md), what was tested locally, and what still
requires a Linux deployment. It is an implementation report, not evidence that
the sandbox is ready for production.

## Current status

The application code is implemented but the feature remains disabled by
default:

```text
SANDBOX_ENABLED=false
```

The sandbox must not be enabled in production until the separate rootless
Podman broker is installed and the deployment checks in this document pass.

A host-side rootless-Podman broker is used in both development and production.
Start the complete development stack with the usual shortcut:

```bash
juena up
```

The launcher starts the rootless Podman machine when needed on macOS, builds the
fixed scientific image, starts the broker as an unprivileged host process, and
then starts the Compose application. The chatbot sends an approved command to
the broker; the broker starts one short-lived Podman container, collects
allowed output files, and removes it. The chatbot receives no Docker or Podman
socket. `juena down` stops the app and broker while preserving data volumes and
leaves the Podman VM warm for the next start. `juena sandbox-stop` remains the
variant that leaves PostgreSQL running.

One real local broker job has generated and returned a Matplotlib PNG. The API
and Streamlit UI have also been started with `SANDBOX_ENABLED=true`, the API
passed its broker health check, and the starter-to-plot flow was exercised in
Streamlit:

```text
Plot with Python starter
    -> software specialist generates a command
    -> Streamlit shows Run / Edit / Cancel
    -> user approves
    -> host rootless-Podman broker executes it
    -> API validates returned files
    -> plot appears in Streamlit and remains downloadable after reload
```

That interactive run exposed a follow-up bug: after the PNG had already been
saved successfully, the specialist tried to print the complete image as Base64
text. This produced a second approval card, leaked a large unreadable string
into the activity panel, and made an internal history write fail on the
read-only container root. The implementation now prevents that redundant
command, hides raw `execute` output from the transcript, ignores the already
handled interrupt when resuming, and stores Deep Agents' internal files under
the writable `/workspace` directory.

A later clean rerun exposed a related backend regression. Deep Agents now
deprecates callable backend factories, so the software specialist receives a
concrete `CompositeBackend`. Its default `RuntimePodmanSandboxBackend` resolves
the authenticated user and thread internally for each operation using the
active LangGraph runtime. Deep Agents can therefore discover both the
execution-capable default and the writable `/workspace` artifact root directly,
without private middleware overrides.

The corrected flow was rerun through the real API on 2026-08-09. It paused on
`python /workspace/outputs/sine_wave_plot.py`, resumed after approval, completed
with exit code `0`, and returned a validated 1482 by 880 PNG plus the reusable
script, generated command, and text output. The raw LangGraph stream emitted the
same initial approval twice; the stream processor now remembers interrupt IDs
and sends each approval card only once.

The macOS machine now has a rootless Podman 6.0.2 Linux VM and the broker path
has been exercised against it. The target production Linux host, hard
filesystem quota, optional gVisor runtime, and the complete agent/UI workflow
through the new Podman path remain unverified.

## Implemented behavior

### Agent and backend

- Only `software-specialist` receives the execution-capable backend.
- The supervisor and neutron specialist remain unable to execute commands.
- A runtime-aware backend instance resolves an isolated identity from the
  authenticated `user_id` and owned `thread_id` for every invocation.
- The sandbox is the default backend required by Deep Agents execution.
- `/inputs`, `/repos`, and `/skills` use explicit read-only routes.
- Uploaded files are copied to the broker's read-only `/inputs` mount.
- Scratch work and generated files use the writable `/workspace` mount.
- `jscatter==1.9.0.1` is baked into the fixed image from the shared
  `repository-requirements.txt` manifest and named in the software specialist's
  runtime-capability prompt.
- ARM64 builds compile jscatter and its native dependencies in a disposable
  builder stage; compilers and package installers do not enter the runtime image.
- There is no `LocalShellBackend` fallback. A broker failure returns an error
  instead of running the command on the JüNA host.

### Broker isolation

The separate broker implementation starts one ephemeral container per job. The
production configuration uses rootless Podman under a dedicated Linux user. An
explicitly opted-in Docker Desktop engine is available for local development
only. Both configurations apply:

- no network;
- a read-only root filesystem;
- a non-root container user;
- all Linux capabilities dropped;
- `no-new-privileges`;
- CPU, memory, PID, output, file-size, and execution-time limits;
- a writable per-user/thread workspace;
- read-only staged inputs;
- no application repository, database, credentials, SAML files, or container
  socket mounted into the job;
- four concurrent execution slots with bounded queue waiting;
- thread deletion and 24-hour idle cleanup.

The broker rejects path traversal and writes to `/inputs`. Artifact collection
also rejects symlinks so a generated link cannot make the broker read a host
file. A real deployment must additionally enforce a hard 5 GB project quota on
each workspace directory; periodic size checks are not a substitute for a
filesystem quota.

The broker drives Podman through the official `podman` (podman-py) library over
the API socket named by `SANDBOX_PODMAN_URI`, rather than by building a CLI
command line. Container supervision is `create` / `start` / `wait` / `logs` /
`remove(force=True)`, so there is no output-reader thread, no polling loop, and
no kill-then-`podman rm` fallback.

Two settings need care because podman-py does not accept the CLI spelling:

- `no_new_privileges=True` is the correct argument. `security_opt` means SELinux
  labels in this library and would silently not apply `no-new-privileges`.
- The user-namespace mode is passed as `{"nsmode": "keep-id", "value":
  "uid=10001,gid=10001"}`. Podman rejects the CLI form `keep-id:uid=...` here.

One tmpfs option cannot be expressed through podman-py's documented arguments.
Its mount renderer forwards only a fixed set of option keys, and `noexec` is not
among them; Podman applies `nosuid` and `nodev` to API-created tmpfs mounts by
default but not `noexec`. `/tmp` therefore requests `noexec` through the one key
whose value is appended verbatim to the mount options. This is asserted by
`test_tmpfs_mount_requests_noexec`, so the protection cannot be dropped silently
by a future podman-py change.

Because the polling loop is gone, workspace size is checked before and after a
job but not during it. The hard 5 GB filesystem quota is consequently required
rather than advisory - see `deploy/SANDBOX.md`.

### Command confirmation

Every agent `execute` request pauses through LangChain's
`HumanInTheLoopMiddleware`. Streamlit displays the exact command and provides:

- `Run`: approve and execute the displayed command with one click;
- `Edit`: reveal the command editor, then require `Run edited command`;
- `Cancel`: reject the command with one click.

The API accepts only the authenticated user's pending interrupt. It rejects
stale or replayed interrupt IDs and edits that attempt to change anything other
than the command text. Pending confirmations are stored by thread and can be
reconstructed from the LangGraph checkpoint after a browser reload.

Approval improves visibility and controls compute use. It is not the security
boundary; the container must remain isolated even when an unsafe command is
approved.

`SandboxExecutionMiddleware` connects the approval to the run. It implements
LangChain's `wrap_tool_call`/`awrap_tool_call`, which sit between the human's
decision and the tool call, and it writes the `generated-command.sh` and
`execution-output.txt` audit records. Because `HumanInTheLoopMiddleware`
interrupts earlier (in `after_model`), reaching this hook means the command was
already approved, and the command it carries is the final, post-edit text. No
approval state is stored, and nothing is matched by comparing command strings.

`execution-output.txt` is saved from the tool's final result text rather than
the raw program output, so it also contains the short note the backend appends
about which artifacts were collected.

### Streaming and status

- LangGraph streaming uses v2 dictionary events.
- Structured `approval_required`, `sandbox_status`, and `artifact` events are
  handled separately from assistant text.
- Subagent reasoning remains hidden from the user-facing answer stream.
- Sandbox queue, completion, timeout, and failure status appears in the existing
  Streamlit activity area.
- Resuming an approved command does not show the same approval card again.
- Raw command output is not repeated as a tool message. The UI receives bounded
  status and validated artifact events instead.
- Internal file-helper operations stay out of the user-facing execution status;
  the status describes the command the user approved.

### Starter prompt

An empty chat shows one additional starter when the sandbox is enabled:

- Label: `Plot with Python`
- It explicitly asks the software specialist to execute NumPy and Matplotlib in
  the sandbox immediately and save both the reusable script and labeled
  sine-wave PNG under `/workspace/outputs`.

This is the existing beginning-of-conversation `st.pills` control. No result
selection chips were added.

Local development defaults `SANDBOX_STARTER_PREVIEW=true`, so the chip remains
visible for interface review when execution is disabled. Clicking a preview-only
chip displays that Python execution is inactive and does not send the prompt.
When `SANDBOX_ENABLED=true`, as in the active local development stack, clicking
the chip sends the plotting prompt. Production rejects preview-only mode and
hides the chip until the real sandbox is enabled.

### Plots and downloads

Generated files under `/workspace/outputs` are imported into persistent,
user-owned artifact storage. The approved command and bounded text output are
also stored automatically.

In plain language: the Python command only needs to save the plot as a PNG in
`/workspace/outputs`. JüNA collects it automatically. The agent must not run a
second command to print, Base64-encode, list, checksum, or re-read that file just
to display it. A backend guard also skips redundant Base64 export commands in
case a model requests one despite the prompt rule.

Streamlit:

- displays validated PNG plots as compact previews below the completed answer;
- removes obsolete inline Base64 image strings from old assistant messages and
  uses the validated stored plot instead;
- opens a larger plot view in a dialog when the user selects `Expand plot`;
- provides a download button for every plot and generated file;
- retrieves historical files through an authenticated API after chat reload;
- never puts raw file data into supervisor/model context.

Allowed downloads are PNG, Python and shell scripts, plain text, logs,
Markdown, CSV, TSV, JSON, notebooks, PDF, and NumPy `.npy`/`.npz` files. HTML,
SVG, native executable binaries, and arbitrary archives are rejected. Non-image
files are served as attachments; only validated PNGs are displayed inline.

Limits are four PNG plots per turn, eight generated files per execution, twelve
files per turn, 5 MB per PNG, and 10 MB per other file. Only PNG bytes travel
inline in SSE; downloadable files are fetched from the authenticated endpoint
to avoid large stream and session-memory usage.

### Ownership, audit, and cleanup

- Artifact paths use opaque IDs under a hash of the authenticated user ID.
- Another authenticated user receives `404` for an artifact they do not own.
- Artifact metadata is attached to the final assistant message so history uses
  the same renderer as a live response.
- Thread deletion removes LangGraph state, the chat row, persistent artifacts,
  and the broker workspace.
- Approval and execution audit records include user, thread, interrupt,
  original/edited command, broker job, outcome, exit code, and artifact IDs.

## Local verification performed

Validated on 2026-08-09 against base revision
`41ce4d69b17d243d97f79c2e2f6e158931d2c796` plus the uncommitted sandbox
worktree. The SHA-256 source snapshot is
`93d6f78bf1607f81dcb4a2b36f89f19d29bc1a537931d59656dc08a052b20314`
(all tracked and untracked project files except `.env*`, runtime-ignored files,
and this verification record).

The following checks were actually run:

| Check | Result |
|---|---|
| Complete repository pytest suite | `304 passed, 4 skipped` |
| Python compilation check | Passed |
| `git diff --check` | Passed |
| Docker Compose configuration without an engine socket or broker container | Passed |
| `juena up` lifecycle | Passed; app and host Podman broker healthy |
| `juena down` graceful shutdown | Passed; app exit `0`, `OOMKilled=false` |
| App-container to host-broker health request | Passed |
| Scientific sandbox image build | Passed on rootless Podman 6.0.2/ARM64 |
| Sandboxed jscatter import | Passed for `jscatter==1.9.0.1`, `fscatter`, and `surface` with no network, a read-only root, and no pip |
| Deterministic `keep-id:uid=10001,gid=10001` workspace write | Passed |
| Real rootless-Podman broker execution and artifact return | Passed (`podman-smoke.txt`) |
| Container network disabled and root filesystem read-only | Passed |
| Output truncation and hard execution timeout | Passed |
| Warm one-container execution time | `0.259 s`, `0.515 s`, `0.255 s` |
| Runtime `pip` unavailable smoke check | Passed |
| Installed environment non-writable smoke check | Passed |
| Local FastAPI startup with sandbox disabled | Passed |
| Local FastAPI startup with sandbox enabled and broker health check | Passed |
| Local Streamlit startup and HTTP response | Passed |

The four skipped tests are existing environment-gated integration tests. They
do not represent successful sandbox deployment tests.

The macOS checks exercise the real rootless-Podman broker, but the Podman VM is
not evidence for target-host filesystem quotas, systemd identity, firewalling,
Linux host mounts, gVisor integration, or production load behavior.

## Not yet executed

The following work remains before production enablement:

1. Install the broker under a dedicated unprivileged account on the target
   Linux host, following [deploy/SANDBOX.md](deploy/SANDBOX.md).
2. Build and inspect the image with rootless Podman on that host.
3. Configure a private/firewalled broker address and the shared random token.
4. Configure and prove a hard 5 GB filesystem/project quota per workspace.
5. Decide whether to enable and test gVisor (`runsc`).
6. Repeat the complete `Plot with Python` agent/UI flow through the new Podman
   path, including approval, inline plot, downloads, chat reload, and thread
   deletion. The earlier interactive flow used the retired Docker path; the
   new Podman acceptance run validated the lower-level broker and artifact path.
7. Prove the job cannot reach the network, host filesystem, application data,
   database, credentials, SAML material, or container sockets.
8. Test approve, edit, reject, timeout, broker outage, replay, cross-user access,
   malformed files, quota exhaustion, and cleanup against the deployed broker.
9. Run the planned load test with 30 active chat sessions and four simultaneous
   sandbox jobs.

Keep `SANDBOX_ENABLED=false` in production until these checks pass. A working
development Podman VM or green unit tests alone are not sufficient evidence to
enable untrusted code execution on shared infrastructure.

## Main implementation locations

| Area | Files |
|---|---|
| Broker, backend, approval, artifacts | `src/juena/sandbox/` |
| Agent integration | `src/juena/agents/specialist_runtime.py`, `src/juena/agents/specialists/software/` |
| API streaming, resume, retrieval | `src/juena/server/api/endpoints.py`, `src/juena/server/streaming/` |
| Streamlit controls and rendering | `app/chat_interface.py`, `app/ui_components.py`, `app/starters.py` |
| Deployment | `deploy/sandbox/`, `deploy/juena-sandbox-broker.service`, `deploy/SANDBOX.md` |
| Configuration | `src/juena/core/config.py`, `env.example`, `docker-compose.yml`, `docker-compose.dev.yml` |
| Tests | `tests/test_sandbox_*.py` and updated client/streaming/UI tests |
