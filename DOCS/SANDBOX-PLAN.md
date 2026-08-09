# JüNA Code Execution Sandbox Plan

Split out of `DEPLOYMENT-AND-SANDBOX-PLAN.md` (now
[DEPLOYMENT-PLAN.md](DEPLOYMENT-PLAN.md)) because the two have different
lifecycles: deployment ships first and changes rarely; the sandbox is a
security-critical subsystem that will iterate on its own.

Content marked **[revised]** differs from the original text and is justified in
[Verified library constraints](#verified-library-constraints).

## Status

The application-side implementation described here is now present in this
repository. It remains **disabled by default** (`SANDBOX_ENABLED=false`) until a
Linux administrator deploys the separate rootless-Podman broker and completes
the isolation and load acceptance tests below.

See [SANDBOX-IMPLEMENTATION.md](SANDBOX-IMPLEMENTATION.md) for the detailed
implementation report, local verification results, and the explicit list of
end-to-end execution checks that have not yet been performed.

Implemented code includes the broker/backend boundary, per-user/thread
workspaces, read-only staged inputs, command approval and resume, v2 streaming,
artifact validation and ownership, inline plots, downloadable generated files,
audit records, cleanup, deployment examples, and focused unit tests. This does
not mean the feature is production-approved: container isolation, host
filesystem quotas, firewall rules, gVisor if selected, and the 30-session load
test must be verified on the actual Linux host.

## Beginner's overview

The goal is to let JüNA calculate things and create plots without letting
model-generated code touch the server that runs JüNA. The code runs in a small,
temporary, heavily restricted container instead.

The intended flow is:

```text
Empty chat offers a "Plot with Python" starter chip
        |
        v
User selects it, or writes a different calculation/plot request
        |
        v
Software specialist prepares a command
        |
        v
JüNA pauses and shows the command to the user
        |
        v
User approves, edits, or rejects it
        |
        v
Sandbox broker starts an isolated container
        |
        v
Container reads only this user's/thread's files and runs without network access
        |
        v
JüNA copies validated results, scripts, output, and plots back into the chat
        |
        v
Temporary container is destroyed
        |
        v
UI shows the stored plot and download buttons below the completed answer
```

The most important security rule is: **user approval is helpful, but the
sandbox is the real protection.** A user may approve a command without noticing
something dangerous. The container must therefore remain safe even when an
unsafe command is approved.

### Terms used in this plan

| Term | Plain-language meaning |
|---|---|
| Agent | The language-model application that decides what action to take. |
| Supervisor | The main JüNA agent that talks to the user and delegates specialist work. |
| Software specialist | The subagent allowed to request sandboxed code execution. |
| Backend | A Python object that tells agent tools where files live and how operations are performed. |
| Sandbox | An isolated environment where untrusted commands can run with strict limits. |
| Container | A short-lived process environment with its own filesystem and resource limits. It is isolation, but not as strong as a separate virtual machine. |
| Broker | A small trusted service that creates, limits, and destroys sandbox containers. |
| Route | A virtual path prefix, such as `/inputs/`, assigned to a particular backend. |
| Mount | A file or directory made visible inside a container. It can be read-only or writable. |
| Checkpointer | Persistent storage that lets an interrupted agent continue from the same point later. |
| SSE | Server-Sent Events, the HTTP stream JüNA uses to send live status and results to the UI. |
| Artifact | A generated file returned to the chat, such as a PNG plot. |
| Egress | Data leaving the sandbox, usually through the network. |
| Thread | One chat conversation and its associated state. |
| Tenant | One authenticated user as an isolated security boundary. Cross-tenant access means one user reaching another user's data. |
| Root filesystem (`rootfs`) | The container's base operating-system and installed-software files. |
| Rootless | Running containers without host administrator (`root`) privileges. |
| Prompt injection | Untrusted text that tries to manipulate the model into taking an unintended action. |
| Base64 | A text representation used to safely transport binary data such as a PNG through JSON/SSE. It is encoding, not encryption. |

## Scope

Give the agent the ability to run scientific Python and shell commands on behalf
of an authenticated FZJ user, return generated plots to the chat, and do so
without granting the model any access to JüNA's infrastructure, other users'
data, or the network.

Not in scope: runtime package installation, GPU workloads, long-lived user
environments, notebook-style incremental sessions.

## Starter conversation chip

Once sandbox execution is available, add one entry to the existing starter
topics in `app/starters.py`:

- **Label:** `Plot with Python`
- **Prompt sent when selected:** explicitly create and execute a NumPy and
  Matplotlib sine-wave script in the sandbox, save both the reusable script and
  PNG under `/workspace/outputs`, and return them through the chat artifact UI.
  It also says not to answer with code alone or encode the image as text.

This uses the existing `st.pills` starter-topic control shown only in an empty
chat. Selecting it sends the prompt immediately as the user's first message and
exercises the complete sandbox, approval, artifact, and plot-rendering flow. It
does not add a new chip beside generated plots. Production keeps this starter
hidden until the sandbox and plot-return path are enabled. Development may show
it with `SANDBOX_STARTER_PREVIEW=true`; while execution is disabled, clicking
the preview shows an explanatory message and does not send the prompt.

## Threat model

The trust question is *not* "is the user malicious." It is: **model-authored code
runs on shared institutional infrastructure, and the human approving it owns
none of that infrastructure.**

With a local coding assistant, the operator usually owns the machine, files, and
consequences, although sandboxing and permission controls are still valuable.
In JüNA, the user, infrastructure owner, and possible blast radius are different
parties. User consent therefore cannot transfer the institution's risk or serve
as the security boundary. Containment must.

Threats, in priority order:

1. **Prompt injection into executable code.** Indexed repository content and
   uploaded files are untrusted. Either can carry text steering the model toward
   code it would not otherwise write. Assume this succeeds; the container is
   what makes it harmless.
2. **Cross-tenant data access.** One user's uploads, memories, or artifacts
   reachable from another user's execution.
3. **Host and infrastructure escape.** Container breakout, socket access,
   credential or SAML material exposure.
4. **Resource exhaustion.** One thread starving the other 29 chat sessions.
5. **Egress.** Exfiltration of uploaded scientific data.

Explicitly accepted: the model may produce *incorrect* scientific results. No
sandbox or approval gate addresses this. It is a UI and prompting concern —
outputs must be presented as agent-generated and unverified.

In simpler terms, this design assumes the generated command will eventually be
wrong or hostile. Security must come from limiting what that command can see,
change, contact, and consume. Correct scientific interpretation is a separate
problem.

## Verified library constraints

Checked against the installed `deepagents==0.6.12` source and cross-checked
against the current LangChain and Deep Agents documentation. The pinned source
is authoritative for implementation details that the public documentation does
not guarantee.

This section is intentionally technical because small library details determine
whether a command actually runs inside the sandbox. The practical conclusions
are: the sandbox must be the default file backend for the software specialist,
ordinary file tools can also enter the sandbox, and the main supervisor should
remain unable to execute commands.

1. **Execution is not path-routable.** `CompositeBackend.execute` delegates to
   `self.default` only and raises `NotImplementedError` unless that default
   satisfies `SandboxBackendProtocol` (`backends/composite.py:535-565`).
   Routing `/workspace/` to a sandbox while keeping a state backend as default
   does not work.
2. **The tool is gated dynamically, not at registration.**
   `FilesystemMiddleware` always registers `execute`, then filters it out of
   `request.tools` on every model call when `supports_execution(backend)` is
   false (`middleware/filesystem.py:1890-1902`; `supports_execution` itself is
   defined at line 653). This is why the model sees no `execute` tool today, and
   why enabling the sandbox needs no tool wiring.
3. **Tool-level permissions and execution are near-exclusive.**
   `FilesystemMiddleware` raises `NotImplementedError` when `_permissions` are
   combined with an execution-capable backend whose paths are not all scoped to
   routes (`middleware/filesystem.py:865-875`). Note the escape hatch:
   `_all_paths_scoped_to_routes` — permissions are usable only when every
   permission pattern starts under an explicit route. That can protect
   `/inputs`, `/repos`, and `/skills`, but cannot cover arbitrary paths handled
   by the execution-capable default backend, such as `/tmp`, `/etc`, or relative
   paths. It therefore cannot close the execution-backend gap on its own.
   Additionally, JüNA assembles specialists with `create_agent`, while Deep
   Agents installs the permission-to-HITL glue in `create_deep_agent`. If direct
   filesystem-tool interrupts are added, JüNA must explicitly install matching
   `HumanInTheLoopMiddleware` rules; passing `_permissions` alone is not enough.
4. **[revised] Inherited file operations reach the container and bypass the
   `execute` tool gate.** `BaseSandbox` (`backends/sandbox.py:586`) implements
   `ls`, `read`, `grep`, and `glob` through `execute()`. `write` delegates to
   `upload_files()`, while `edit` uses `execute()` for small payloads and a mix
   of uploads and execution for large payloads. Its docstring states it "does
   not reduce or partition the trust boundary of `execute()`." Consequence:
   `HumanInTheLoopMiddleware(interrupt_on={"execute": ...})` leaves six
   unapproved filesystem-tool paths into the container. Shell-backed argument
   construction is injection-safe (`shlex.quote` and encoded payloads), so this
   is not a shell-injection hole — but it means **the container must be safe
   without approval.**
5. **[revised] `BaseSandbox` reduces the build to four methods.** Abstract
   surface is exactly `execute`, `upload_files`, `download_files`, and `id`.
   The original plan's framing implied writing a filesystem backend; it does
   not. Everything else is inherited.

Documentation references: [Deep Agents sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes),
[backends](https://docs.langchain.com/oss/python/deepagents/backends),
[permissions](https://docs.langchain.com/oss/python/deepagents/permissions), and
[LangChain human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop).

## Backend composition

Constraint 1 forces the sandbox to be the default and everything safe to be an
explicit route:

Think of `CompositeBackend` as a path switchboard. When an agent requests
`/repos/project/file.py`, the switchboard sends that request to the read-only
repository backend. A path without a matching route goes to `default`, which is
the sandbox. This is why choosing the default is security-sensitive.

```
CompositeBackend(
    default = PodmanSandboxBackend(...),      # /workspace, satisfies execute
    routes  = {
        "/inputs/":   <read-only route-local adapter over staged uploads>,
        "/repos/":    ReadOnlyFilesystemBackend(repo cache),
        # Present only when software-specialist owns a valid SKILL.md.
        "/skills/":   ReadOnlyFilesystemBackend(software skills),
    },
)
```

This inverts the current topology into a **fail-open** shape: today the default
is `ReadOnlyInputsStateBackend` and unmatched paths fail closed; afterwards
every unmatched path lands in the execution backend.

Here, **fail-open** does not mean the command escapes the container. It means a
path that the routing code does not recognize is still handled by the sandbox
instead of being rejected. The sandbox must therefore safely handle every
possible unmatched path.

Two consequences:

- The `/inputs` read-only guard currently lives in `ReadOnlyInputsStateBackend`
  ([backends.py](src/juena/agents/backends.py)) — the class that is the
  specialist default today. Once the sandbox is the default, `/inputs` needs an
  explicit read-only route or staged uploads silently become writable. Do not
  mount `ReadOnlyInputsStateBackend` there unchanged: `CompositeBackend` strips
  the matched `/inputs/` prefix before delegation, so that class would neither
  find the existing state keys nor recognize them as read-only. Use a dedicated
  route-local adapter that restores the sandbox `/inputs` prefix and rejects all
  mutation. Stage the same validated upload manifest into the sandbox as a
  read-only `/inputs` mount so file tools and shell commands see identical data.
- The full route table must be passed to `CompositeBackend` at construction.
  It caches a longest-prefix-first `sorted_routes` table in `__init__`; mutating
  `routes` later does not update that table and is unsupported.

### Runtime ownership

The compiled JüNA graph is process-global and reused across users. Therefore the
software specialist must receive a **backend factory**, not one static
`PodmanSandboxBackend` instance. `FilesystemMiddleware` supports a callable
backend resolved from `ToolRuntime`; use its trusted runtime context to require
both `user_id` and `thread_id`, then construct or retrieve the corresponding
composite backend.

The broker key is `(user_id, thread_id)`, never `thread_id` alone. A caller
presenting an existing thread ID under another authenticated user must be
rejected before sandbox lookup. The backend factory must also be used by every
specialist middleware that accesses the filesystem so they all resolve the same
workspace for one invocation.

The workspace is a broker-managed volume scoped to that key and retained across
turns for up to 24 hours of inactivity. Containers are ephemeral per sandbox
job, including shell-backed `BaseSandbox` helpers, and mount that workspace plus
the read-only input tree. Thread deletion immediately destroys the workspace;
idle expiry is the fallback cleanup path.

The `(user_id, thread_id)` pair is comparable to a storage locker key. Using
only `thread_id` would be unsafe because another authenticated user might guess
or obtain that value. Both values must come from trusted server-side
authentication/runtime data, never directly from an editable request body.

**[revised] Test requirement.** A spot check on `/inputs` is not enough for a
fail-open default. The test must enumerate every path prefix the model can name
— including `/`, `/tmp`, `/etc`, unprefixed relative paths, `..` traversal, and
the `artifacts_root` prefixes `FilesystemMiddleware` derives
(`/large_tool_results`, `/conversation_history`) — and assert each resolves to
the intended backend.

**[revised] Decision: `execute` belongs to `software-specialist`, not the
supervisor.** The topology inversion is unavoidable, but it need not affect the
agent handling conversation, memory, and final synthesis. Scoping the sandbox
composition to the specialist keeps the supervisor's memory-only backend
fail-closed and confines the fail-open blast radius to delegated software work.
The supervisor and each specialist already own separate backends. Per LangChain
guidance, the checkpointer stays on the **top-level** agent only; subagent
interrupts propagate to it.

## Sandbox backend

- Subclass `deepagents.backends.sandbox.BaseSandbox`; it is not re-exported from
  `deepagents.backends` in `0.6.12`. Implement `execute`, `upload_files`,
  `download_files`, and `id`. Never use `LocalShellBackend` in production.
- **[revised]** Enforce that in code, not documentation: assert at startup that
  the production backend satisfies `SandboxBackendProtocol` and is not
  `LocalShellBackend`, and fail hard otherwise. Do not add a bypass flag —
  Claude Code's `dangerouslyDisableSandbox` is the anti-pattern here.
- Run a dedicated rootless Podman service under an unprivileged `juena-sandbox`
  account. Only the broker may reach its socket; FastAPI and agent containers
  receive neither Docker nor Podman sockets.
- Per-authenticated-user/thread workspace volume; ephemeral container per
  sandbox job, as defined in [Runtime ownership](#runtime-ownership).
- Image: Python 3.11, NumPy, SciPy, pandas, matplotlib, seaborn, SymPy,
  scikit-learn, Pillow, basic shell utilities, and pinned repository-backed
  software (initially `jscatter==1.9.0.1`). No runtime package installation.

Development and production use the same host-side rootless-Podman broker and
the same container command. On macOS, Homebrew Podman connects to its rootless
Linux VM; on production Linux, systemd runs the broker as the dedicated
`juena-sandbox` account. The application reaches the broker over HTTP and never
receives a Docker or Podman socket. Platform-specific launchers differ, but the
execution policy, data path, concurrency, and resource limits do not.

### Isolation

Isolation controls are the sandbox's actual safety boundary. For example,
`Network: none` prevents uploaded data from being sent to an external server;
`Rootfs: read-only` prevents changes to the container's installed software; and
the CPU, memory, process, and timeout limits prevent one job from exhausting the
whole JüNA server.

A read-only root filesystem does not make every path read-only. The dedicated
`/workspace` volume remains writable so calculations can create temporary files
and plots, while `/inputs` and the installed software remain read-only.

| Control | Value |
|---|---|
| Concurrency | 4 jobs; excess queues |
| Per job | 2 vCPU, 4 GB RAM, 256 PIDs, 1 GB tmpfs |
| Timeout | 10 min hard |
| Workspace | 5 GB per thread, 24 h idle expiry |
| Upload | 100 MB per turn |
| Rootfs | read-only |
| User | non-root |
| Capabilities | all dropped, `no-new-privileges` |
| Seccomp | default confinement |
| Network | none |

`vCPU` is a limited share of processor time, RAM is working memory, and PIDs
limit how many processes a job may create. Dropping Linux capabilities removes
special operating-system powers; `no-new-privileges` prevents a process from
gaining more privilege later; and seccomp blocks dangerous system calls at the
kernel boundary.

Never mount application repositories, database data, credentials, or SAML
material. Copy accepted user files in as read-only inputs. Truncate excessive
stdout/stderr before returning it to the model.

**[revised] Consider gVisor (`runsc`) as the Podman runtime.** The deployment
plan accepts "a weaker shared-kernel boundary" as residual risk. gVisor
interposes a user-space kernel on syscalls and drops in as a runtime without
architectural change; overhead is syscall-bound and negligible for matplotlib
workloads. This is the cheapest retirement of the stated residual risk.

In plain language, ordinary containers still share the host's Linux kernel.
gVisor adds another layer between an untrusted command and that kernel. A
microVM gives stronger separation but needs more infrastructure, while a
browser-style WebAssembly environment is more restrictive and cannot provide
the required shell behavior.

Alternatives considered: Firecracker microVMs (true hypervisor boundary,
disproportionate ops weight at 4 concurrent jobs); Pyodide/WASM (no kernel
surface, supports the entire listed scientific stack, but no shell and no
package flexibility); managed services such as E2B or Modal (ruled out by FZJ
data residency).

## Human-in-the-loop

Human-in-the-loop means JüNA pauses before the `execute` tool runs and waits for
a decision:

- **Approve:** run the command exactly as shown.
- **Edit:** change only the command text, then run the changed command.
- **Reject:** do not run it; return the rejection to the agent.

Streamlit presents these as `Run`, `Edit`, and `Cancel` buttons. `Run` and
`Cancel` take one click. `Edit` first reveals a command field, then requires a
separate `Run edited command` click. The panel is stored by thread and can be
reconstructed from the LangGraph checkpoint after a browser reload.

The pause improves visibility and controls compute usage. It does not make an
unsafe container safe, because file tools can still perform sandbox operations
and people can approve dangerous-looking commands by mistake.

**[revised] What this gate is for.** Per constraint 4, approval covers `execute`
but not the six inherited file operations, so it is not the security boundary.
Given the isolation table above — no network, no secrets, read-only rootfs,
ephemeral, per-thread — its genuine functions are:

- **Comprehension.** The user sees what will run before it runs.
- **Compute cost.** Prevents runaway jobs consuming the 4-slot queue.

It is *not* preventing exfiltration, host compromise, or wrong results. Stating
this matters for UX: block-by-default on every execution produces approval
fatigue within a week, and rubber-stamped approval is worse than no gate — it
manufactures false assurance and creates an audit trail implying the user vetted
something they did not. Budget UI effort toward *showing results* over
*collecting consent*.

**Open question:** given the above, should low-cost pure computation
auto-approve, with the gate reserved for long-running or resource-heavy jobs?
Claude Code's tiered model (safe-command classification plus explicit allow
rules) is the reference. Deferred — ship block-by-default, instrument approval
rates, revisit with data.

### Mechanism

- `HumanInTheLoopMiddleware(interrupt_on={"execute": ...})`, with allowed
  decisions declared explicitly and a `description_prefix` naming the sandbox
  limits. It requires a checkpointer, already present via Postgres.
- `FilesystemPermission(mode="interrupt")` is optional defense in depth for
  explicitly routed filesystem paths only. Because JüNA uses `create_agent`,
  wire the corresponding filesystem-tool HITL predicates explicitly if this is
  enabled. Do not treat those private Deep Agents helpers as a stable public API
  or as containment for the default sandbox backend.
- Add an authenticated resume SSE endpoint. A single pending interrupt resumes
  with `Command(resume={"decisions": [...]})`; multiple pending interrupts use
  `Command(resume={interrupt_id: {"decisions": [...]}, ...})`.
- Stream the resumed graph with `agent.astream(..., version="v2")`. In the
  installed LangGraph, v2 emits dictionaries shaped like
  `{type, ns, data, interrupts}`, while JüNA's existing processor consumes v1
  tuples. Update `StreamEventProcessor` and its tests as part of this work; do
  not pass v2 events into the current tuple parser.
- Reject stale, replayed, cross-thread, and cross-user decisions.
- **[revised] The `edit` decision carries `edited_action` with both `name` and
  `args`, both client-controllable.** Enforce server-side that `name` is
  unchanged and that timeout, resource policy, mounts, network policy, and
  workspace identity are not editable. Only the command text may change. A
  UI-only restriction is not a control.

### Streamed events

| Event | Payload |
|---|---|
| `approval_required` | interrupt ID, command, description, limits, allowed decisions |
| `sandbox_status` | queued, running, completed, timed out, rejected |
| `artifact` | artifact ID, filename, MIME type, size, caption; inline PNGs also include base64 |

## Artifacts and downloads

An artifact is an output file that should survive after the temporary container
is gone. It may be a plot, script, text output, table, notebook, PDF, or NumPy
data file. Generated files are not trusted merely because they came from the
sandbox. JüNA first downloads them through the broker, checks their type and
size, stores them under the authenticated user and thread, and only then sends
them to the UI.

- User-facing files are written to `/workspace/outputs`. The approved command
  and its bounded stdout/stderr are also saved as downloads automatically.
- Once the approved command succeeds, the agent stops executing. It must not
  run `ls`, `cat`, Base64, checksum, or other follow-up commands merely to make
  the UI find an output. The application already collects the output directory.
  A backend guard skips redundant Base64 export attempts as defense in depth.
- Inline plots: PNG only, at most 4 per turn, 5 MB each, with bounded pixel
  dimensions.
- Downloads: at most 8 files per execution and 12 total files per turn. Each is
  at most 10 MB. Allowed extensions are `.py`, `.sh`, `.txt`, `.log`, `.md`,
  `.csv`, `.tsv`, `.json`, `.ipynb`, `.pdf`, `.npy`, `.npz`, and `.png`.
  Active or executable web formats such as HTML and SVG are rejected.
- Emit artifact events directly from the specialist execution path and queue
  them for the active turn. Only bounded inline PNGs use base64 in SSE; other
  files are fetched through the authenticated endpoint. Raw base64 never enters
  the supervisor or model context; the specialist report contains artifact IDs
  and metadata only.
- Associate persisted artifact metadata with the completed assistant turn so
  live streaming and chat reload use the same rendering path.
- Streamlit decodes validated queued payloads, renders plots directly beneath
  the completed supervisor answer, and provides a download button for every
  generated file. It does not render artifacts while specialist reasoning runs.
- The resumed interrupt ID is filtered from the response stream so an approval
  that the user just handled cannot immediately appear again. Raw `execute`
  tool output is not rendered in the transcript or activity preview.
- Persist metadata by authenticated user and thread. Authenticated retrieval for
  chat reloads; delete files when the thread is deleted.

## Audit

Record per execution: user, thread, interrupt ID, original and edited commands,
decision, sandbox job ID, timestamps, exit status, artifact references.

The audit record answers who requested the operation, what was approved, what
actually ran, and what it produced. It should contain identifiers and bounded
command/output metadata, not duplicate large uploaded scientific datasets.

## Suggested implementation order

Each stage should be tested before moving to the next one. This keeps failures
small and makes the security boundary easier to review.

1. Build and test the route-local read-only `/inputs` adapter without enabling
   execution.
2. Define the sandbox broker interface and use a fake sandbox backend in unit
   tests. Verify user/thread isolation and route resolution first.
3. Implement the rootless Podman backend and independently prove its network,
   mount, secret, filesystem, and resource restrictions.
4. Connect the per-runtime backend factory to the software specialist. Confirm
   that the supervisor and neutron specialist still have no `execute` tool.
5. Add `HumanInTheLoopMiddleware`, authenticated resume handling, v2 stream
   parsing, replay prevention, and ownership checks.
6. Add artifact validation, persistence, SSE events, direct Streamlit plot
   rendering, and authenticated file downloads for live responses and chat
   history. Then enable the `Plot with Python` starter prompt.
7. Run the end-to-end, cleanup, security, concurrency, and load acceptance tests
   before enabling the feature in production.

## Acceptance

Acceptance criteria are the tests and observable results required before this
feature can be considered ready. A working plot alone is not enough; isolation,
ownership, cleanup, and failure behavior must also pass.

- Sandbox jobs cannot reach the network, application volumes, environment
  secrets, container sockets, PostgreSQL, or the host filesystem.
- **[revised]** Exhaustive route-resolution test (see
  [Backend composition](#backend-composition)), not a single `/inputs` case.
- Backend-factory tests require authenticated `user_id` and `thread_id`, reject
  cross-user reuse of a thread ID, reuse one workspace for the same identity,
  and isolate all different identity pairs.
- `/inputs` tests verify route-prefix stripping is restored by the adapter,
  mutations fail, and file tools and shell commands observe the same staged
  bytes.
- **[revised]** Startup assertion rejects a non-sandbox backend in production.
- **[revised]** Server-side rejection of an `edited_action` whose `name` differs
  from the original or whose arguments change anything except command text.
- Resume-stream tests cover v2 dictionary events, multiple interrupt IDs, and
  supervisor/subagent namespaces without regressing the existing answer filter.
- Approve, edited approval, rejection, timeout, replay prevention, ownership
  enforcement, queue saturation, workspace cleanup, thread deletion.
- Starter-topic tests verify that `Plot with Python` appears only when sandbox
  execution is enabled and sends the defined example prompt as the first user
  message.
- Artifact retrieval tests prove that a user cannot display another user's plot
  or download another user's generated file by changing an artifact ID, thread
  ID, or event payload.
- End-to-end matplotlib test: approved specialist Python creates a PNG, the API
  queues validated artifact data, Streamlit shows the image directly below the
  completed supervisor answer, and it remains available after chat reload
  without rerunning the script. Run this test from the `Plot with Python`
  starter prompt as well as through a normal typed request.
- Download test: the approved command, bounded text output, generated script,
  and requested data files remain downloadable after chat reload. Unsupported
  active formats are rejected.
- 30 active chat sessions with 4 simultaneous sandbox jobs; bounded memory,
  stable PostgreSQL pools, predictable queueing.

## Open questions

1. Tiered auto-approval for cheap computation — deferred pending approval-rate
   data.
2. gVisor now, or ship on stock Podman and add `runsc` once load-tested?
