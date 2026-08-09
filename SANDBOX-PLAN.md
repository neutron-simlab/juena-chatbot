# JüNA Code Execution Sandbox Plan

Split out of `DEPLOYMENT-AND-SANDBOX-PLAN.md` (now
[DEPLOYMENT-PLAN.md](DEPLOYMENT-PLAN.md)) because the two have different
lifecycles: deployment ships first and changes rarely; the sandbox is a
security-critical subsystem that will iterate on its own.

Content marked **[revised]** differs from the original text and is justified in
[Verified library constraints](#verified-library-constraints).

## Scope

Give the agent the ability to run scientific Python and shell commands on behalf
of an authenticated FZJ user, return generated plots to the chat, and do so
without granting the model any access to JüNA's infrastructure, other users'
data, or the network.

Not in scope: runtime package installation, GPU workloads, long-lived user
environments, notebook-style incremental sessions.

## Threat model

The trust question is *not* "is the user malicious." It is: **model-authored code
runs on shared institutional infrastructure, and the human approving it owns
none of that infrastructure.**

This is the structural difference from a local coding agent such as Claude Code,
which needs no sandbox because its operator, approver, and blast radius are the
same person. In JüNA they are three different parties, so user consent transfers
no risk and cannot serve as the boundary. Containment must.

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

## Verified library constraints

Checked against the installed `deepagents==0.6.12` source, not documentation.

1. **Execution is not path-routable.** `CompositeBackend.execute` delegates to
   `self.default` only and raises `NotImplementedError` unless that default
   satisfies `SandboxBackendProtocol` (`backends/composite.py:535-565`).
   Routing `/workspace/` to a sandbox while keeping a state backend as default
   does not work.
2. **The tool is gated dynamically, not at registration.**
   `FilesystemMiddleware` always registers `execute`, then filters it out of
   `request.tools` on every model call when `supports_execution(backend)` is
   false (`middleware/filesystem.py:653`). This is why the model sees no
   `execute` tool today, and why enabling the sandbox needs no tool wiring.
3. **Tool-level permissions and execution are near-exclusive.**
   `FilesystemMiddleware` raises `NotImplementedError` when `_permissions` are
   combined with an execution-capable backend whose paths are not all scoped to
   routes (`middleware/filesystem.py:865-875`). Note the escape hatch:
   `_all_paths_scoped_to_routes` — permissions *are* usable if every permission
   path maps to a route. Worth testing before abandoning
   `FilesystemPermission(mode="interrupt")`, which `deepagents>=0.6.8`
   supports and which would gate filesystem tools that
   `HumanInTheLoopMiddleware` on `execute` cannot reach (see next point).
4. **[revised] Inherited file operations are container execution, and bypass the
   `execute` gate.** `BaseSandbox` (`backends/sandbox.py:586`) implements `ls`,
   `read`, `write`, `edit`, `grep`, and `glob` by calling `execute()`. Its
   docstring states it "does not reduce or partition the trust boundary of
   `execute()`." Consequence: `HumanInTheLoopMiddleware(interrupt_on={"execute":
   ...})` leaves six unapproved paths into the container. Argument construction
   is injection-safe (`shlex.quote` in `_build_grep_cmd`, base64 payloads in
   `_build_ls_cmd`/`_build_glob_cmd`), so this is not a shell-injection hole —
   but it means **the container must be safe without approval.**
5. **[revised] `BaseSandbox` reduces the build to four methods.** Abstract
   surface is exactly `execute`, `upload_files`, `download_files`, and `id`.
   The original plan's framing implied writing a filesystem backend; it does
   not. Everything else is inherited.

## Backend composition

Constraint 1 forces the sandbox to be the default and everything safe to be an
explicit route:

```
CompositeBackend(
    default = PodmanSandboxBackend(...),      # /workspace, satisfies execute
    routes  = {
        "/inputs/":   <read-only guard over staged uploads>,
        "/repos/":    ReadOnlyFilesystemBackend(repo cache),
        # Present only when software-specialist owns a valid SKILL.md.
        "/skills/":   ReadOnlyFilesystemBackend(software skills),
    },
)
```

This inverts the current topology into a **fail-open** shape: today the default
is `ReadOnlyInputsStateBackend` and unmatched paths fail closed; afterwards
every unmatched path lands in the execution backend.

Two consequences:

- The `/inputs` read-only guard currently lives in `ReadOnlyInputsStateBackend`
  ([backends.py](src/juena/agents/backends.py)) — the class that is the
  specialist default today. Once the sandbox is the default, `/inputs` needs an
  explicit read-only route or staged uploads silently become writable.
- `CompositeBackend` aggregates route state schemas at construction, so the full
  route table must be passed up front. Routes cannot be added later.

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

- Subclass `BaseSandbox`; implement `execute`, `upload_files`, `download_files`,
  `id`. Never use `LocalShellBackend` in production.
- **[revised]** Enforce that in code, not documentation: assert at startup that
  the production backend satisfies `SandboxBackendProtocol` and is not
  `LocalShellBackend`, and fail hard otherwise. Do not add a bypass flag —
  Claude Code's `dangerouslyDisableSandbox` is the anti-pattern here.
- Run a dedicated rootless Podman service under an unprivileged `juena-sandbox`
  account. Only the broker may reach its socket; FastAPI and agent containers
  receive neither Docker nor Podman sockets.
- Per-thread workspace; ephemeral container per execution.
- Image: Python 3.11, NumPy, SciPy, pandas, matplotlib, seaborn, SymPy,
  scikit-learn, Pillow, basic shell utilities. No runtime package installation.

### Isolation

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

Never mount application repositories, database data, credentials, or SAML
material. Copy accepted user files in as read-only inputs. Truncate excessive
stdout/stderr before returning it to the model.

**[revised] Consider gVisor (`runsc`) as the Podman runtime.** The deployment
plan accepts "a weaker shared-kernel boundary" as residual risk. gVisor
interposes a user-space kernel on syscalls and drops in as a runtime without
architectural change; overhead is syscall-bound and negligible for matplotlib
workloads. This is the cheapest retirement of the stated residual risk.
Alternatives considered: Firecracker microVMs (true hypervisor boundary,
disproportionate ops weight at 4 concurrent jobs); Pyodide/WASM (no kernel
surface, supports the entire listed scientific stack, but no shell and no
package flexibility); managed services such as E2B or Modal (ruled out by FZJ
data residency).

## Human-in-the-loop

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

- `HumanInTheLoopMiddleware(interrupt_on={"execute": ...})`, with a
  `description_prefix` naming the sandbox limits. Requires a checkpointer —
  already present via Postgres.
- Authenticated resume SSE endpoint resuming the LangGraph checkpoint with
  `Command(resume={"decisions": [...]})` and `version="v2"`. Decisions are keyed
  by interrupt ID when several are open.
- Reject stale, replayed, cross-thread, and cross-user decisions.
- **[revised] The `edit` decision carries `edited_action` with both `name` and
  `args`, both client-controllable.** Enforce server-side that `name` is
  unchanged and that resource policy is not editable. A UI-only restriction is
  not a control.

### Streamed events

| Event | Payload |
|---|---|
| `approval_required` | interrupt ID, command, description, limits, allowed decisions |
| `sandbox_status` | queued, running, completed, timed out, rejected |
| `artifact` | artifact ID, filename, MIME type, size, base64 payload |

## Artifacts

- Plots written to `/workspace/outputs`; validate and import PNGs into
  persistent artifact storage after execution.
- Limits: 4 images per execution, 5 MB each, bounded pixel dimensions. Reject
  malformed data and active formats such as SVG.
- Emit artifact events directly from the specialist execution path and queue
  them for the active turn. Raw base64 never enters the supervisor or model
  context; the specialist report contains artifact IDs and metadata only.
- Streamlit decodes validated queued payloads and renders them beneath the
  completed supervisor answer, rather than while specialist reasoning runs.
- Persist metadata by authenticated user and thread. Authenticated retrieval for
  chat reloads; delete files when the thread is deleted.

## Audit

Record per execution: user, thread, interrupt ID, original and edited commands,
decision, sandbox job ID, timestamps, exit status, artifact references.

## Acceptance

- Sandbox jobs cannot reach the network, application volumes, environment
  secrets, container sockets, PostgreSQL, or the host filesystem.
- **[revised]** Exhaustive route-resolution test (see
  [Backend composition](#backend-composition)), not a single `/inputs` case.
- **[revised]** Startup assertion rejects a non-sandbox backend in production.
- **[revised]** Server-side rejection of an `edited_action` whose `name` differs
  from the original.
- Approve, edited approval, rejection, timeout, replay prevention, ownership
  enforcement, queue saturation, workspace cleanup, thread deletion.
- End-to-end matplotlib test: approved specialist Python creates a PNG, the API
  queues validated artifact data, Streamlit renders it below the completed
  supervisor answer, and it remains available after chat reload.
- 30 active chat sessions with 4 simultaneous sandbox jobs; bounded memory,
  stable PostgreSQL pools, predictable queueing.

## Open questions

1. Tiered auto-approval for cheap computation — deferred pending approval-rate
   data.
2. gVisor now, or ship on stock Podman and add `runsc` once load-tested?
3. Does `_all_paths_scoped_to_routes` let `FilesystemPermission(mode="interrupt")`
   coexist with the sandbox? If so, it closes the constraint-4 gap that
   `HumanInTheLoopMiddleware` cannot.
4. Workspace persistence across turns within a thread is assumed. Confirm
   against the 24 h idle expiry and thread-deletion semantics.
