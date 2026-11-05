# Foreground steering and Stop

While a reply runs, the message box says **Guide this response**. Text submitted
there reaches each active specialist before its next model call and reaches the
supervisor before synthesis. A message marked **Applied** has been included in a
model request; it does not guarantee that the model followed the instruction.
An instruction arriving too late is marked **Not applied** and can be copied back
to the message box. It never silently starts another turn and is excluded from
later model requests and summaries unless the user resubmits it.

**Stop** cancels the foreground graph and its sandbox commands. **Stopping** lasts
until cleanup is confirmed. Completed file changes remain. Detached background
research is outside this control. Text such as "abort last prompt" is an ordinary
steering instruction; use the Stop button for deterministic cancellation.

## Release requirements

The API, UI and sandbox worker must contain the same `juena-core` revision. The
feature first ships in core `1c76ca1`, pinned in `pyproject.toml`; do not deploy
this chatbot against an older core pin.

The API must run in a single process. Live controls are process-local and are
pruned after an hour or when the terminal-run cache fills as new runs start. Applied messages
and final delivery statuses are stored in the existing conversation checkpoints.
Unapplied instructions use a separate transcript-only checkpoint channel. Saving
delivery receipts preserves pending clarification/approval task and interrupt IDs.
The control object is runtime context, never graph state. No new service is needed.

The UI requires Streamlit 1.64 or newer. Its session-owned reader performs HTTP
I/O without Streamlit calls, while a short fragment refresh renders events.
Closing a session, switching conversations or logging out closes that connection
and triggers the same cancellation cleanup on the API.

## Migration and coordinated restart

Drain foreground replies and sandbox commands before changing versions. Do not
run old and new sandbox workers concurrently. Back up the database using the
normal deployment procedure. No database reset is required.

For local development, stop the app and worker while retaining PostgreSQL:

```sh
./juena sandbox-stop
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T postgres \
  sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < deploy/steering-and-stop.sql
./juena up
```

For the standard VM deployment using `/etc/juena/juena.env`:

```sh
sudo ./juena prod down
sudo docker compose -p juena-chatbot -f docker-compose.yml \
  --env-file /etc/juena/juena.env up -d postgres
sudo docker compose -p juena-chatbot -f docker-compose.yml \
  --env-file /etc/juena/juena.env exec -T postgres \
  sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < deploy/steering-and-stop.sql
sudo ./juena prod up
```

Use the deployment's actual environment file if it differs from this default.
The production launcher stops the dedicated worker and imports the running app's
exact image before restarting it. Retain that existing flow.

The migration adds nullable `sandbox_jobs.run_id`, a default-false
`cancel_requested` flag, and an index. It is safe to apply repeatedly and does
not change existing results. Queued cancellation becomes terminal immediately;
running cancellation retains capacity and workspace protection until the worker
removes its container. Late results cannot overwrite cancellation or attach
artifacts. Worker recovery preserves cancellation requests.

## Acceptance and diagnostics

After deployment, use a fresh authenticated browser session to start a research
reply, submit a narrowing instruction while it runs, and confirm one final answer
reflects the correction. Repeat with Stop during model/tool waiting and with a
running sandbox command delegated to the software specialist. Before Stop,
confirm `sandbox_jobs.run_id` matches the foreground run ID. After Stop, confirm
the job is `cancelled`, `cancel_requested` is true, the owned container is gone,
and the next prompt works. A direct backend check does not cover this delegation
boundary. Also exercise clarification/approval resume,
ordinary uploads, and artifacts. The local Podman test does not replace VM acceptance.

Inspect logs by run ID and instruction ID. Control logs report acceptance,
application recipients and terminal status without recording instruction text or
credentials. Unconfirmed graph or sandbox termination retains the conversation
reservation; inspect sandbox rows and container cleanup before restarting services.
A history-save failure is reported separately and releases the reservation once
execution has ended.

Authenticated API controls are `POST /threads/{thread_id}/runs/{run_id}/steer`
(JSON `instruction_id`, `text`), `POST .../stop`, and `GET` of that run URL.
Foreign conversations return 404; a second foreground invocation returns 409.
Stale run IDs never select another invocation. New instructions are text only.
The status GET is available for diagnostics and recovering missed status events;
the UI normally receives status on the existing response stream.

For browser regression without provider calls, start the fixture:

```sh
uv run python -m streamlit run \
  tests/steering_browser_app.py --server.address 127.0.0.1 --server.port 19621 \
  --server.headless true --browser.gatherUsageStats false
```

In another terminal, with Playwright installed (the sibling core explorer's
development dependencies can supply it):

```sh
NODE_PATH=../juena-core/frontend/filesystem/node_modules node tests/browser_steering.cjs
```

The fixture binds an in-memory API to localhost:19622. It checks the real message
box, steering during research, Stop during a stalled tool, a fresh next turn and
steering followed by clarification and approval in the same reply. It uses no
provider credentials or production database; its approval tool executes no command.
