# Langfuse tracing

JüNA can send every agent run to a Langfuse server of your own. The UI is at
http://localhost:3000. It shows every model call, tool call and specialist
run, grouped by conversation (*session*) and by user. Token counts appear
only when the model provider reports them. Blablador's streamed replies
currently report none.

Langfuse runs as its own Compose project (`deploy/langfuse/`), separate from
the chatbot's. `./juena down` leaves it running, so the traces stay viewable
between chatbot restarts.

LangSmith is still available. The two have independent switches,
`LANGFUSE_TRACING` and `LANGSMITH_TRACING`. If both are on, every run goes to
both.

## Development

```bash
./juena langfuse-up       # first start: about 2-4 minutes (database migrations)
```

The first `langfuse-up` creates `deploy/langfuse/.env` from `env.example`.
That file holds the login for http://localhost:3000
(`LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`).

Then copy the tracing block from `env.example` into the chatbot's `.env`:

```bash
LANGFUSE_TRACING=true
LANGFUSE_BASE_URL=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=pk-lf-juena-development-only
LANGFUSE_SECRET_KEY=sk-lf-juena-development-only
```

Then run:

```bash
./juena langfuse-check    # sends a test trace with those keys and reads it back
./juena up                # warns if the chatbot container cannot reach Langfuse
```

`LANGFUSE_BASE_URL` depends on where JüNA runs:

| JüNA runs as | `LANGFUSE_BASE_URL` |
|---|---|
| containers (`juena up`) | `http://langfuse:3000` |
| native (`juena run`) | `http://localhost:3000` |

Other commands:

| Command | What it does |
|---|---|
| `juena langfuse-logs` | follow the web and worker logs |
| `juena langfuse-down` | stop Langfuse; traces, login and keys are kept |
| `juena langfuse-reset --confirm` | **delete** every trace, the login and the keys |

## How the pieces connect

- **Key pairs.** The chatbot's key pair must equal
  `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `_SECRET_KEY` in `deploy/langfuse/.env`.
  Both example files start with the same development pair. `langfuse-check`
  proves they match.
- **Network.** The chatbot container reaches Langfuse as `langfuse` over a
  private Docker network, `juena-tracing`, which the launcher creates. Only the
  Langfuse web container joins it. Its databases are on Langfuse's own network
  and are not reachable from the chatbot.
- **Published ports.** Only the UI and API are published, and only on
  `127.0.0.1:3000`. Langfuse's Postgres, ClickHouse, Redis and MinIO publish
  nothing, so nothing clashes with the chatbot's development Postgres on 5432.
  This works the same way on macOS, WSL and Linux.

## What a trace is

- **One graph invocation is one trace.** That means a new message, the answer
  to an `ask_user` question or a sandbox approval, or one background research
  job.
- **Grouping.** All of a conversation's traces share one session (the chat's
  thread id) and the user's id. That includes its research jobs. Langfuse's
  Sessions and Users pages are built from that.
- **Attribution.** Every observation carries the session and user, not only
  the root. That is what makes per-user and per-session token totals add up.
  `tests/test_langfuse_tracing.py` checks it on the exported spans.

## What gets lost, and when

- **Langfuse down.** Chats keep working. The SDK buffers spans in memory and
  retries for a while, but it does not store them on disk. Traces produced
  during a longer Langfuse outage can be lost.
- **Graceful stop.** When the chatbot stops (`juena down`, SIGTERM), these
  steps all fit within the 30 s stop grace period:
  1. Open chat streams end at once, and each one first finishes cancelling
     its graph, for up to 5 s.
  2. Other in-flight requests get up to 10 s.
  3. Research is stopped.
  4. The buffered spans are flushed.

  A run cut off this way keeps its trace, including the graph and node steps.
  A research job shows ERROR-level observations for the cut. One thing
  is missing: the model call that was mid-answer at that moment. LangChain's
  agents call the model in a way that never reports a cancelled call as
  ended, so Langfuse never receives that one generation.
- **Forced kill.** Spans still in the buffer are lost.
- **Contents.** Traces contain full chat text, staged inputs and tool output.
  Access control is the Langfuse login.

## Resources

ClickHouse, Postgres, Redis, MinIO, and the Langfuse web and worker containers
use about 2-3 GB of memory when idle, and ClickHouse peaks higher. Docker
Desktop's VM needs room for that next to the chatbot (`JUENA_MEMORY_LIMIT`).

## Before exposing it beyond this machine

The development stack is loopback-only on purpose. Its secrets and its
login/API key pair are well-known values from `env.example`.

- **Use a fresh deployment** with generated secrets (`openssl rand -hex 32`).
  Do not widen an existing development instance.
- **`LANGFUSE_INIT_*` only creates resources that do not exist yet.** Editing
  them after the first start rotates nothing.
- **If you must reuse an instance,** first change the user's password and
  revoke the development API keys in the Langfuse UI. Changing `SALT`
  invalidates every API key.

## Kubernetes notes

For when JüNA moves to a cluster. Nothing here is built yet.

- **Chart.** Use the official chart,
  `oci://ghcr.io/langfuse/langfuse-k8s/charts/langfuse`. Pin the version;
  `2.1.5` was the latest on 2026-10-13. It needs Helm ≥ 3.17 and
  Kubernetes ≥ 1.28.
- **Cluster prerequisites, once per cluster,** when the chart deploys its own
  ClickHouse: **cert-manager** and the **ClickHouse operator** with its CRDs
  (`clickhouseclusters.clickhouse.com`, `keeperclusters.clickhouse.com`). The
  chart's README has the exact install commands and versions.
- **Object storage.** The chart's bundled store is **SeaweedFS**, not MinIO.
- **Secrets.**
  - Supply `langfuse.salt`, `langfuse.encryptionKey` and
    `langfuse.nextauth.secret` through `secretKeyRef`.
  - Pass the `LANGFUSE_INIT_*` values through `langfuse.additionalEnv`, from a
    Secret.
- **Datastores.** Start with the bundled ones. Postgres can move to
  CloudNativePG later (`postgresql.deploy: false` plus host and auth), the
  same operator juena-rag uses.
- **Ingress.** Add one with TLS. On the chatbot side only `LANGFUSE_BASE_URL`
  changes, to the in-cluster Service.
- **OpenShift.** Check the security context constraints for components that
  run as fixed user IDs.
- **Chatbot replicas.** Langfuse's own replica count is a separate decision.
  On the chatbot side, each process keeps its tracing state in memory, so
  scaling it out needs nothing tracing-specific. Re-check the background
  research recovery design before running more than one chatbot replica.
