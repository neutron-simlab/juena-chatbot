# Sandbox cleanup plan

Status: proposed, not started.

---

## What this is about

The sandbox lets the chatbot write a small program and actually run it — to make a plot, do some
statistics, read a data file. Running code that a language model wrote is risky, so the code runs
inside a locked-down container: no network, no writing outside its own folder, a memory and time
limit, and no administrator rights.

The setup we built for this works. But while reviewing it, two parts turned out to be written by
hand when a ready-made, well-maintained tool already does the same job. This document describes
replacing those two parts.

**Nothing about the security model changes.** The container stays just as locked down. What
changes is *how we tell the computer to start it*, and *how we remember that a human clicked
"Run"*.

---

## The two problems

### Problem 1: We wrote our own remote control for Podman

Podman is the program that starts and stops containers. Right now, `broker_service.py` talks to
Podman the way a person would type at a terminal: it builds a long text command, runs it as a
child process, and then babysits that process itself.

Concretely, the code currently does all of this by hand:

- Builds a list of ~20 command-line flags. One of them is inserted with
  `command[6:6] = [...]`, meaning "squeeze this into position 6 of the list". If anyone ever adds
  or reorders a flag above that line, this silently puts the wrong thing in the wrong place.
- Starts a **separate background thread** just to read the program's output as it arrives, and
  passes a one-item list `[False]` between threads as a makeshift "was the output too long?" flag.
- Runs a **loop that wakes up 4 times per second** to check two things: has the time limit passed,
  and has the folder grown too big? The size check scans every file in the whole workspace folder,
  every quarter second, for every job that is running.
- If the time runs out, it kills the process, *and then also* runs a separate `podman rm -f`
  command in case the first kill did not clean up.

There is an official Python library from the Podman project itself — the `podman` package
(also called podman-py, version 5.8.0, maintained at `containers/podman-py`). It does all of the
above as normal function calls:

- `containers.create(...)` — start a container, with each setting as a named argument
- `container.wait(timeout=600)` — wait for it to finish, or give up after 10 minutes
- `container.logs()` — read the output
- `container.remove(force=True)` — clean up, reliably, once

No hand-built text command. No background thread. No polling loop. No cleanup-after-the-cleanup.

### Problem 2: Approval and execution are connected by comparing text

When the chatbot wants to run a command, a card appears asking the human to approve, edit, or
cancel it. When the human clicks Run, that decision needs to reach the part of the code that
actually runs the command — so it can show "queued…" and "finished", and save an audit record of
what was approved.

Right now this is done by writing the approved command into a shared dictionary in memory
(`ArtifactStore._approved_commands`), and then, at execution time, **searching that dictionary
for a command whose text matches exactly**.

This is fragile, and it fails quietly. If the two pieces of text differ by even one space, the
lookup finds nothing — and the code treats "not found" as "this wasn't approved". But it **still
runs the command anyway**. It just skips the progress messages and skips saving the audit record.
So the worst case is: the command runs, and we lose our record that a human approved it. That is
the opposite of what an audit trail is for.

LangChain already has the right tool for this: a hook called `wrap_tool_call`. It sits directly
between "the human approved" and "the command runs", and it receives the approved command as a
direct argument. Nothing to store, nothing to look up, nothing to compare.

---

## What we are deliberately *not* doing right now

**Files are sent between the two programs as text-encoded blobs (base64 inside JSON).** Every
uploaded data file and every generated plot gets converted into a long string of letters, sent
over HTTP, and converted back. This is about 150 lines of the broker plus 5 data definitions, and
it exists only because the two programs talk over the network.

They don't have to. The container's workspace is just an ordinary folder on the same machine.
If both programs could see that folder directly, all of this conversion code would disappear.
But that ties the two programs to the same server and moves some safety checks around, so it is a
separate decision for another day.

Two smaller findings are also recorded here so they are not forgotten:

1. **Artifacts are queued in plain Python dictionaries held in memory.** That works only because
   the server runs as a single process. Running two copies of the server for more capacity would
   break plot delivery, because each copy would have its own separate dictionary.
2. Some smaller things: the same "create-this-once" pattern copy-pasted in two places, a new HTTP
   connection opened and thrown away on every call instead of reusing one, and the approval card
   showing hardcoded text `"2 vCPU"` / `"4 GB"` that is not connected to the real limits — if
   someone changes the actual limit, the card keeps showing the old number.

---

## Part 1: Use the official Podman library

**Main file:** `src/juena/sandbox/broker_service.py`
**Also:** `pyproject.toml`, everything in `deploy/`, the `juena` launcher script,
`tests/test_sandbox_broker.py`

### Step 1 — Check that nothing gets lost in translation (do this first)

This is the one genuine risk, so it goes before any rewriting.

Every safety setting is currently a command-line flag. The library uses named arguments instead.
Most map over cleanly, but **if one of them quietly does not work, we would lose a safety
protection without any error message appearing.** That is not acceptable for security settings.

So: write a throwaway script that starts one container using the library, then reads back what
Podman *actually* configured (via `container.attrs`) and checks each setting is really there.
Pay particular attention to:

- `userns_mode="keep-id:uid=10001,gid=10001"` — this is how files created inside the container
  stay readable and writable outside it
- the tmpfs options `nosuid,nodev,noexec` — these stop programs being run out of the temporary
  folder
- `ulimits` — the maximum size of any single file

If any of these do not survive the round trip, keep using the command line for that one setting
and write down why.

### Step 2 — Translate the settings

Two of these do **not** work the way the command-line name suggests. Both were verified against
the library's own source:

| Command line today | Library argument | Note |
|---|---|---|
| `--network=none` | `network_mode="none"` | |
| `--read-only` | `read_only=True` | |
| `--cap-drop=all` | `cap_drop=["ALL"]` | |
| `--security-opt=no-new-privileges` | `no_new_privileges=True` | ⚠️ **not** `security_opt` — that argument means something else in this library (SELinux labels) |
| `--cpus=2` | `cpu_period=100000, cpu_quota=200000` | ⚠️ there is **no** `cpus` argument; CPU limits are expressed as a fraction |
| `--memory=4g` | `mem_limit="4g"` | |
| `--pids-limit=256` | `pids_limit=256` | |
| `--ulimit=fsize=N:N` | `ulimits=[{"Name": "fsize", "Soft": N, "Hard": N}]` | |
| `--tmpfs=/tmp:rw,nosuid,...` | `tmpfs={"/tmp": "rw,nosuid,nodev,noexec,size=1g"}` | |
| `--volume=<ws>:/workspace:rw,Z` | `mounts=[{"type": "bind", ...}]` | |
| `--volume=<inputs>:/inputs:ro,Z` | same, plus `"read_only": True` | |
| `--userns=keep-id:uid=...` | `userns_mode="keep-id:uid=10001,gid=10001"` | the `command[6:6]` trick disappears here |
| `--workdir` `--env` `--name` `--rm` | `working_dir` `environment` `name` `remove=True` | |
| `--runtime=runsc` | `runtime=...` | only if gVisor is enabled |

### Step 3 — Connect to Podman

Instead of pointing at the `podman` program on disk, we point at Podman's socket — a small file
that acts like a phone line to Podman.

- Replace the `SANDBOX_PODMAN_BINARY` setting with `SANDBOX_PODMAN_URI`, defaulting to
  `unix:///run/user/<uid>/podman/podman.sock`.
- Open one connection when the service starts and close it when it stops (the existing `lifespan`
  function already does this kind of setup and teardown).
- The startup self-check keeps its three existing safety checks (refuse to run as administrator,
  require a long enough token, create the workspace folder). The two places where it currently
  shells out to `podman` become `client.ping()` and `client.images.exists(...)`.

### Step 4 — Rewrite the run loop

Everything about *what* happens stays the same: the limit on how many jobs run at once, the
per-workspace lock, recreating the outputs folder, and comparing before/after to find new files.
Only the supervision changes:

```python
container = self._create_container(request, container_name)
try:
    container.start()
    try:
        exit_code = container.wait(timeout=request.timeout_seconds)
        timed_out = False
    except (APIError, ReadTimeoutError):
        timed_out, exit_code = True, None
    output, truncated = self._read_logs(container, request.max_output_bytes)
finally:
    with suppress(APIError, NotFound):
        container.remove(force=True)
```

`_read_logs` reads the output as it streams in and stops once it hits the size limit — replacing
the background thread and the shared `[False]` flag.

**Deleted:** the reader thread, the 4-times-per-second loop, the manual kill, and the
`podman rm -f` fallback.

### ⚠️ One real consequence, stated plainly

That polling loop was doing a second job: checking every quarter second whether the workspace
folder had grown past 5 GB. Removing the loop means we check the total size **before** a job and
**after** a job, but not **during** it.

This is the right way round, and both existing documents already say so. `deploy/SANDBOX.md`
states that the periodic check *"cannot replace a filesystem quota while an untrusted process is
actively writing"* — in other words, the check was never actually reliable, because a program can
write faster than we can measure.

The real protections are the per-file size cap (`ulimit fsize`, unchanged) and a proper
filesystem quota enforced by the operating system.

**So: the 5 GB filesystem quota changes from "strongly recommended" to "required".** Update
`deploy/SANDBOX.md` to say this clearly, with the reason.

### Step 5 — Deployment updates

- `deploy/juena-sandbox-broker.service` — add the socket path as an environment variable.
- `deploy/SANDBOX.md` — add one setup command (`systemctl --user enable --now podman.socket` for
  the `juena-sandbox` account), and promote the filesystem quota to a required step.
- `deploy/sandbox-broker.env.example` — swap `SANDBOX_PODMAN_BINARY` for `SANDBOX_PODMAN_URI`.
- The `juena` launcher script — **this is the most likely place to hit friction.** On macOS,
  Podman runs inside a small virtual machine, so the socket is not at the standard Linux path.
  Check what `podman machine inspect` reports and set the URI from that.

### Step 6 — Fix a weak test

`tests/test_sandbox_broker.py:74` currently checks that certain flag *strings appear somewhere* in
the command list. That test would still pass with the flags in completely the wrong order — so it
never actually verified much.

Replace it with a test that checks the named arguments handed to `containers.create`, one setting
at a time. This is the test that would catch a setting silently not working, which the old one
could not do. The path-traversal and file-collection tests are unaffected — leave them alone.

---

## Part 2: Connect approval to execution properly

**Main files:** `src/juena/sandbox/middleware.py`, `backend.py`, `artifacts.py`, `approvals.py`,
`src/juena/agents/specialist_runtime.py`, and the matching tests

### The idea

Add `SandboxExecutionMiddleware` to `src/juena/sandbox/middleware.py`, next to the existing
`ArtifactMessageMiddleware`.

The approval card is raised by `HumanInTheLoopMiddleware`, which pauses things at an earlier stage
(`after_model`) than where tools actually run. So by the time `wrap_tool_call` is reached, the
human has **already** decided. If the command arrives here at all, it was approved — and the
command text it carries is the final, post-edit version.

That single fact removes the need for the shared dictionary, the text comparison, and the whole
possibility of the two drifting apart.

```python
def wrap_tool_call(self, request, handler):
    if request.tool_call.get("name") != "execute":
        return handler(request)
    if not requires_execution_approval(request):   # reuse policy.py as-is
        return handler(request)
    # Getting here means the human already approved or edited this command.
    ...save generated-command.sh from request.tool_call["args"]["command"]...
    result = handler(request)
    ...save execution-output.txt...
    return result
```

Write both the normal and the `async` version (`wrap_tool_call` and `awrap_tool_call`). In
practice only the async one runs, but `ArtifactMessageMiddleware` already sets this precedent.

Register it in `specialist_runtime.py` under the same `if enable_execution_approval:` condition as
the approval card itself, so runs that never needed approval do not get approval-related records.

### What gets deleted

- `artifacts.py` — the `_approved_commands` dictionary and its two methods
  (`mark_approved_command`, `consume_approved_command`)
- `approvals.py` — the one call that wrote into that dictionary. Everything else stays, including
  the important rule that editing can only change the command text and never the timeout or the
  folder mappings.
- `backend.py` — the lookup and every `if interrupt_id is not None:` branch around it

`PodmanSandboxBackend.execute` still sends the "queued" and "finished" progress messages, but now
**always**, without asking whether the command was approved. It is a sandbox run either way, and
the backend is the only place that knows the exit code.

### One small behaviour change

`execution-output.txt` will now be saved from the tool's final result text rather than the raw
program output, so it also contains the short note we append about which files were collected.
For an audit record that is arguably better, but it *is* a change — note it in
`SANDBOX-IMPLEMENTATION.md`.

### Tests

Move the "these three files get saved" expectation from the backend test to a new middleware test.
Remove the two approval tests that check the dictionary behaviour, and keep everything else.

Then add the test the old design **could not** pass:

> When a human **edits** the command before approving, the saved `generated-command.sh` contains
> the **edited** text, not the original.

---

## How to check it all works

1. `uv sync` after adding `podman>=5.8.0` to `pyproject.toml`.
2. **Run the settings probe from Part 1, Step 1.** Do not start rewriting until it passes, or
   until a fallback has been chosen for whatever failed.
3. `uv run pytest`. The starting point is `304 passed, 4 skipped`. The total will shift as tests
   move between files — that is fine, but no test should be deleted unless its check reappears
   somewhere else.
4. Then try it for real, with `SANDBOX_ENABLED=true` and `./juena up`:
   - Ask for a plot → approve → progress messages appear, the image shows up, and
     `generated-command.sh` and `execution-output.txt` are saved.
   - Ask again, but **edit** the command before approving → the edited version is what runs, and
     what gets saved. *(This is the case the old design got wrong.)*
   - Run `sleep 700` → it times out, and `podman ps -a` shows **no** leftover container. This is
     the part that proves the new cleanup works as well as the old kill-plus-fallback.
   - Try writing a file larger than 5 GB → it fails on the per-file limit, confirming that
     protection still holds without the polling loop.
5. Confirm the existing check still passes: one user cannot download another user's files
   (the endpoint returns 404). Untouched by this work, but it is the tenant-isolation guarantee.

---

## Still outstanding, unchanged by this work

The 9-item **"Not yet executed"** list in `SANDBOX-IMPLEMENTATION.md` is not affected here:
installing on a real Linux host, the hard filesystem quota, the gVisor decision, the
network/socket/database escape proofs, and the 30-session load test.

`SANDBOX_ENABLED=false` should stay the production default until those are done.
