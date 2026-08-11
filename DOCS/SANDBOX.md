# JüNA sandbox and specialist execution audit

This document is the authoritative audit and remediation record for JüNA's
specialist execution path. It covers the supervisor, specialist agents,
Deep Agents middleware, approval flow, Postgres job channel, rootless-Podman
worker, artifact store, server streaming, and final user-visible response.

Operational installation remains in
[`deploy/SANDBOX.md`](../deploy/SANDBOX.md). This document records what is
implemented, what the August 2026 incident proved, what it did not prove, and
which controls must pass before production enablement.

## Executive verdict

The audited false JScatter fit and PNG claim was not caused by the sandbox
executing successfully or by the artifact store losing a registered file. The
sandbox retained a failed command, returned a nonzero result, and registered no
result artifact. The false claim was created above that boundary:

1. The software specialist ended with an incomplete progress sentence:
   `Now let me check the jscatter skill documentation:`.
2. The pinned Deep Agents subagent middleware accepted the last non-empty
   assistant message as the delegated task result.
3. The supervisor treated that text as a successful handoff and invented
   JScatter APIs, numerical fit values, a PNG filename, and a visual
   description without execution or artifact evidence.
4. The server streamed the supervisor's answer because it had no typed
   specialist-completion or evidence contract to reject it.

This is a critical result-integrity defect at the agent/server contract. It
does not, by itself, demonstrate a Podman isolation, Postgres lifecycle, or
artifact-copying defect.

The remediation policy is serial and fail closed. At most one specialist task
may run for one user turn. A specialist result is user-visible only after the
server verifies its structured completion state, required execution status,
and required result artifacts. Skill usage is left to the model's lazy
relevance decision. A single verified report is returned directly without
supervisor rewriting.

## Audit boundary and evidence

The audit inspected the checked-out source, pinned `deepagents==0.6.12`
implementation, specialist and supervisor prompts, middleware order, server
stream filtering, sandbox job records, authenticated artifact state, persistent
workspace contents, and representative saved chat threads.

### Latest JScatter incident

Thread `1a240294-6c00-41b2-b12b-4223d74b8d60` asked for a JScatter SAS fit and
plot.

| Layer | Observed evidence | Verdict |
| --- | --- | --- |
| Specialist | First delegation returned only the progress sentence quoted above | Incomplete handoff accepted |
| Supervisor | Answer used nonexistent `js.models.Sphere`, `js.Data`, and `js.Fit` APIs and supplied invented fit values | Ungrounded synthesis |
| Initial execution | No sandbox execution preceded the first claimed result | Claim had no execution evidence |
| Follow-up jobs | Fourteen job records existed and none registered a result artifact | Job control retained the absence |
| Failed command | `AttributeError: module 'jscatter' has no attribute 'models'` | Nonzero execution was visible |
| Artifact store | No `sas_fit.png` artifact existed | Claimed image was not delivered |
| Workspace | `/workspace/sas_fit_script.py` existed; `sas_fit.png` did not | Persistent workspace agreed with the artifact store |
| Skill behavior | Skill metadata was available, but some branches inspected repositories or executed without first reading it | Prompt-only gate was bypassed |

The specialist later correctly observed that the PNG was absent, but
incorrectly said the workspace did not persist between executions. Workspaces
are persistent per authenticated user and thread; containers are ephemeral per
command.

### Later three-helper follow-up

A user-supplied Claude Code audit records a later attempt that exposed four
additional failure modes in the same workflow:

1. The supervisor split one JScatter objective among three helpers: one read
   examples and wrote a script without running it, one searched for the output,
   and one received only the script and was told to execute it. The executing
   helper did not receive the evidence used by the writing helper.
2. The first script used SasView-style `js.models.Sphere`, `js.Data.from_file`,
   and `js.Fit` calls rather than JScatter's `js.ff.sphere` and `data.fit` API.
   It failed immediately with `AttributeError`.
3. The executing helper probed the installed package, derived a corrected API,
   and attempted to replace `/workspace/sas_fit_script.py`. Deep Agents'
   `write_file` create-only behavior rejected the existing path, so the
   corrected script was not saved before the run ended.
4. Repository paths under `/repos/...` are virtual routes exposed by read
   tools, not paths mounted inside execution containers. Skill wording did not
   make that boundary explicit, so `cd /repos/...` and executing repository
   examples failed. Separately, JScatter fits can start xmgrace progress
   rendering; `js.headless(True)` must be called before fitting, not merely
   before an explicit plotting call.

The key conclusion is narrower than "the model could not fix the API": the
executing helper did correct it, but fragmented delegation withheld its source
context and the scratch-file lifecycle prevented the correction from replacing
the failed script. These are orchestration and tool-contract defects, not
evidence that the corrected JScatter calculation itself failed.

### Skill-decision loop after the first remediation

The first remediation added a blocking runtime gate and a synthetic
`skill_decision` tool. A later live trace emitted the same no-match result
eleven times: `Skill gate recorded: no available skill applies to this
objective.` This showed that the gate had converted progressive disclosure
into a compulsory protocol. The model repeatedly satisfied the protocol even
though no skill was relevant.

[Deep Agents' skill contract](https://docs.langchain.com/oss/python/deepagents/skills)
leaves relevance to the model: metadata is always available, while `SKILL.md`
and references are read only when the model judges them useful. Therefore skill
selection is not an execution-truth boundary and must not block repository,
documentation, filesystem, or sandbox tools. The correct control is passive
observation of successful exact `SKILL.md` reads, with no no-match tool and no
outcome failure when the model uses no skill.

### Specialist todo loop after script creation

A later software-specialist trace completed `write_file` for a requested plot
script, then called `write_todos` repeatedly without reaching `execute`. This
was possible because every specialist was explicitly given LangChain's
`TodoListMiddleware`. That middleware injects both the planning tool and strong
instructions to update it after each step, but its guard only rejects multiple
parallel todo calls in one model response; it does not stop the same update
from recurring across model turns.

The specialists already receive one serial, scoped objective and must finish
with a typed report. Todo state is neither execution evidence nor report
evidence. The correct control is therefore to omit `TodoListMiddleware` from
specialists, making `write_todos` unavailable there. The supervisor retains its
planning middleware for pre-delegation conversation and routing.

### Surviving successful JScatter thread

Thread `738e6d06-3bb9-4828-8f69-642ba32d436b` demonstrates that the underlying
pipeline can work: later exit-zero jobs registered `initial_plot.png` and
`multicontrastfit.png`. Earlier jobs in that thread failed on invalid Grace
usage and a missing `gracebat`. This is evidence for pipeline capability, not
evidence that every agent answer is grounded or that the target production host
is ready.

### Historical evidence

The previously reported thread `e39cc824-c29f-4c2d-8699-3405d659475a` is no
longer present in the current database. Earlier observations about that thread
must therefore be treated as historical rather than reproducible current-state
evidence.

## End-to-end truth ownership

```text
Authenticated user request
    |
    v
Supervisor creates a typed specialist contract
    |
    v
Specialist optionally reads a relevant skill through progressive disclosure
    |
    v
Human approval for software-specialist execute calls
    |
    v
Postgres sandbox_jobs admission and host worker claim
    |
    v
Ephemeral networkless rootless-Podman container
    |
    v
Persistent per-user/thread workspace and changed output scan
    |
    v
Authenticated artifact registration and SSE delivery
    |
    v
Specialist outcome validator compares report with direct evidence
    |
    v
Root finalizer returns the verified report or a fail-closed limitation
```

Truth is split deliberately:

- The model proposes work and explains scientific meaning.
- `SkillsMiddleware` exposes metadata and the model decides relevance; a passive
  observer records successful exact skill reads without blocking other work.
- The sandbox backend proves command completion and exit status.
- The artifact store proves which files were accepted and made deliverable.
- Outcome middleware proves that the specialist contract was satisfied.
- The root finalizer controls what becomes the final user-visible answer.

Model prose is never execution or artifact evidence.

## Findings

### Critical: incomplete specialist result accepted

The pinned subagent middleware falls back to the last non-empty assistant
message when no structured response exists. A progress sentence can therefore
be returned as if it were a completed delegated task. Model-call limits and
other early termination paths can produce the same condition.

Required control: every specialist must produce a validated structured outcome;
missing or incomplete outcomes are failures.

### Critical: supervisor can invent completion

Specialist tokens and reports are hidden from the normal user answer stream.
The supervisor receives free-form task text and writes the final response. No
server boundary currently prevents it from changing a failure into success or
describing an artifact that does not exist.

Required control: verified single-specialist reports bypass supervisor
rewriting. Invalid results fail closed. Multiple task calls in one user turn
are rejected before any specialist starts.

### High: parallel helpers fragment evidence and correction state

Delegating parts of one objective to different helpers breaks the assumption
that the process reading the skill and examples is also the process writing,
probing, correcting, and validating the script. Passing only generated code to
another helper discards the evidence needed to audit that code.

Required control: allow exactly one specialist task per user turn. A request
that genuinely requires different specialist types must be split into separate
user turns rather than fanned out by the supervisor.

### High: runtime skill enforcement breaks progressive disclosure

`SkillsMiddleware` supplies progressive metadata so the model can decide
whether a skill is useful. Making every repository, Context7, filesystem, and
execution call depend on a skill-read or no-match event changes that design
into an unconditional gate. The repeated no-match trace demonstrates the
result: extra tool calls and looping for an objective that needs no skill.

Required control: keep `SkillsMiddleware` as the source of dynamic metadata;
let the model read relevant skills lazily; expose no no-match tool; never fail a
specialist solely because it did not read a skill. Record an exact successful
`SKILL.md` read passively for audit evidence only.

### High: delegation lacks evidence requirements

The current task tool accepts a free-form description. It does not state
whether execution is required or which result artifacts must exist, so task
completion cannot be evaluated mechanically.

Required control: typed delegation fields for objective, specialist,
`requires_execution`, and required artifact kind/name.

### High: nonzero execution may carry success metadata

Deep Agents' filesystem tool can return a `ToolMessage` whose content contains
`[Command failed with exit code 1]` while its message status remains `success`.
Checking `ToolMessage.status` is therefore insufficient.

Required control: preserve typed exit evidence from the sandbox backend and
mark nonzero or unavailable execution as an error at the local middleware
boundary.

### Medium: specialist failures are hidden by streaming

The server intentionally hides subagent tokens and task reports. This keeps the
UI clean but makes the root final answer the only visible truth boundary. The
current boundary has no structural validation.

### Medium: prompts and skills are process-cached

The compiled agent graph is cached for the service lifetime. Prompt and skill
changes do not become active until the service restarts. Validation that does
not restart may exercise stale resources.

### Medium: regression coverage stops at metadata and prompts

The test suite checks skill discovery and prompt language but did not reproduce
the complete failure chain: ignored skill, incomplete handoff, supervisor
fabrication, and absent artifact.

### Medium: virtual repository and scratch-file contracts were ambiguous

The software prompt and JScatter skill named `/repos/...` paths without saying
that they are available only through read tools. They also did not warn that
`write_file` refuses an existing scratch path. Both omissions turned a
recoverable model correction into another failed run.

Required control: forbid `/repos` in shell commands, imports, subprocesses, and
script paths; use `/workspace` for executable scratch state; and revise an
existing file with `edit_file` or a new versioned filename.

### Medium: JScatter headless mode must precede fitting

JScatter can invoke xmgrace for fit-progress rendering even when a script does
not construct a final plot explicitly. Enabling headless mode only at plotting
time is too late.

Required control: call `js.headless(True)` immediately after import and before
every fit or `GracePlot` construction. The sandbox image must retain `gracebat`
for native headless plot output.

## Verified sandbox architecture

Postgres `sandbox_jobs` is the sole application-to-worker control channel. The
chatbot receives no Podman socket. The host worker runs generated commands in
fresh rootless containers with no network, a read-only root filesystem,
dropped capabilities, `no-new-privileges`, bounded resources, and controlled
mounts.

`/workspace` is writable and persistent for one opaque user/thread workspace;
`/inputs`, `/repos`, and `/skills` are read-only routes where configured.
Only changed, policy-accepted files under `/workspace/outputs` can become result
artifacts. Record artifacts such as the approved command and captured output
are stored separately and must never satisfy a requested result-artifact
requirement.

Human approval remains compute consent and command visibility, not the host
security boundary. The shared Linux kernel, host capacity, rootless runtime,
filesystem permissions, database topology, and optional gVisor configuration
must still be validated on the production Linux host.

## Target invariants

1. A user turn can launch at most one specialist task; parallel task calls are
   rejected before any helper runs.
2. Skill metadata is injected dynamically; the model decides whether to load a
   skill from its description.
3. No synthetic no-match decision is required, and skills never block unrelated
   repository, documentation, filesystem, or execution work.
4. A successful exact `SKILL.md` read may be recorded for audit evidence but is
   not a completion prerequisite.
5. A progress sentence cannot satisfy a delegated task.
6. A failed or absent execution cannot satisfy `requires_execution`.
7. A filename or image description cannot satisfy an artifact requirement;
   only an authenticated result-artifact reference can.
8. Earlier failures are tolerated only when a later execution satisfies every
   requirement in the same task.
9. Any malformed or unmet outcome produces a deterministic limitation rather
   than another model completion.
10. A verified single-specialist report is returned without supervisor
   rewriting.
11. `/repos` is never used as an execution-container path; executable scratch
    files live under `/workspace` and existing files are revised explicitly.
12. JScatter headless mode is enabled before any fit or GracePlot construction.
13. Evidence state survives summarization, approval resume, and chat reload.
14. Prompt and skill validation restarts the service before live testing.
15. Specialists do not expose `write_todos`; a delegated objective proceeds
    directly through relevant tools to its structured report.

## Remediation ledger

| Work item | Final status | Validation |
| --- | --- | --- |
| Documentation-first audit | Complete | This document records evidence, non-findings, controls, and deployment gates |
| Serial specialist boundary | Complete | Parallel current-turn task calls run no specialist; legacy multi-task checkpoints also fail closed |
| Shared dynamic skill inventory | Complete | `SkillsMiddleware` injects the specialist-owned inventory and paths |
| Model-chosen progressive disclosure | Complete | No no-match tool exists; unrelated tools are never blocked; exact reads are observed passively |
| Specialist todo-loop removal | Complete | Specialist middleware exposes no `TodoListMiddleware` or `write_todos`; supervisor planning remains available |
| Typed specialist task contract | Complete | Execution and artifact requirements are Pydantic-validated fields |
| Structured specialist report | Complete | Both specialists use completed, blocked, or failed structured output |
| Direct execution evidence | Complete | Backend status, exit code, result artifact IDs, and dropped outputs survive middleware flattening |
| Outcome validation | Complete | Missing report, required execution, artifact, or delivery evidence downgrades the task to failed; skill use is not required |
| Fail-closed root finalization | Complete | The supervisor cannot rewrite a delegated result; verified reports render directly and invalid reports produce deterministic limitations |
| Regression reproduction | Complete | The exact progress-sentence plus fabricated fit/PNG chain is prevented without calling the prepared supervisor synthesis |
| JScatter execution-path guidance | Complete | `/repos`, headless fitting, and scratch-file revision contracts are explicit and regression-tested |
| Scientific `.dat` delivery | Complete | `.dat` fit results are collected only after UTF-8 validation and are reloadable as `text/plain` |
| Live JScatter/Grace smoke | Complete at sandbox truth boundary | Foreground worker completed a weighted sphere fit, native Grace PNG, result registration, claim, and persisted reload |
| Live external-model acceptance | Pending operator smoke | A real Blablador/Qwen turn must confirm relevant skills load lazily without repeating no-match actions |
| Target-Linux production gate | Blocked | Capacity and isolation checks must pass on the deployment host |

## Validation gates

Implementation validation must include focused middleware and server tests, the
complete non-integration suite, Postgres integration through
`./juena test-integration`, Compose rendering, compilation, packaging, and a
foreground-worker live run. The live run must restart the application first and
cover successful execution, rejection, nonzero exit, missing required output,
artifact reload, workspace exclusion, and cleanup.

The JScatter smoke must verify the installed version, construct
`js.GracePlot(headless=True)` without the missing-Gracebat exception, fit data
with uncertainties, write a PNG below `/workspace/outputs`, and deliver the
registered artifact. A neutron-specialist request must also demonstrate the
structured direct-report path without execution requirements.

The current macOS development host cannot close the production Linux capacity
or isolation gate. Production must keep `SANDBOX_ENABLED=false` until the
operational checklist in `deploy/SANDBOX.md` passes on the actual host.

## Final validation evidence

Validation on 2026-08-11 produced the following current-state results:

- JScatter skill `quick_validate.py`: valid.
- Complete pytest suite after removing runtime skill enforcement: `363 passed,
  14 skipped`.
- Postgres integration launcher: an earlier stopped-worker run passed all 9;
  the final run passed 8 and skipped only the worker-lock test because the
  normal development worker was deliberately running.
- Opt-in live Postgres/Podman pipeline: `3 passed, 1 skipped` before the
  JScatter-specific addition; the new JScatter test then passed independently.
- JScatter live run: imported and printed `1.9.0.1`; called
  `js.headless(True)` before fitting; completed an uncertainty-weighted sphere
  fit with a finite radius error; constructed `GracePlot(headless=True)`;
  registered `.dat`, `.json`, and `.png` result artifacts; claimed them at the
  application boundary; and reloaded the persisted PNG by authenticated ID.
- Compose development rendering, Python compilation, and `git diff --check`:
  passed.
- Wheel and source distribution: built successfully. The wheel contains both
  agent prompts, passive skill-usage observation, typed contracts, finalizers, execution-evidence
  module, JScatter `SKILL.md`, `agents/openai.yaml`, and all four references.
- Rebuilt application container: healthy after repository bootstrap; the first
  immediate health request raced startup, while the subsequent container health
  check and log confirmed HTTP 200 readiness.

The foreground validation worker was stopped after the smoke test to release
its single-worker advisory lock. The rebuilt application, Postgres, and normal
background sandbox worker were all healthy and running at the final state
check. Do not start a second worker while that normal worker owns the lock.
