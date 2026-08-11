# JüNA Production Deployment Plan

The implemented code-execution architecture and its remaining production gates
live in [SANDBOX.md](SANDBOX.md). This document covers only what the sandbox
needs from the deployment: a host-side rootless-Podman worker, shared Postgres,
and disk for disposable workspaces.

## Summary

Deploy JüNA as one Docker Compose application containing several single-purpose containers—not one oversized container with multiple processes.

Production target:

- Proxmox VM running Ubuntu Server 24.04 LTS.
- 10–30 concurrent chat users.
- Institutional reverse proxy owns DNS and HTTPS.
- Recommended VM: 16 vCPU, 32 GB RAM, approximately 500 GB SSD/NVMe.
- No GPU required.
- Maximum four concurrent scientific Python/shell sandboxes.
- One production command: `docker compose --env-file /etc/juena/juena.env up -d`.
- Runtime configuration comes exclusively from Compose/environment files; users never configure or see a backend URL in the chatbot.

## VM and Infrastructure Prerequisites

- Allocate approximately 60 GB for the OS, 300 GB for persistent JüNA data, and 140 GB for disposable sandbox workspaces.
- Install Docker Engine with Compose v2, rootless Podman for sandbox execution, `qemu-guest-agent`, Git, NTP, and standard monitoring utilities.
- Permit inbound traffic only:
  - SSH from the administration network.
  - One HTTP gateway port, recommended `8080`, from the institutional proxy CIDR.
  - Optional monitoring ports from the monitoring network.
- Do not publish PostgreSQL, FastAPI, Streamlit, or Podman ports externally.
- Allow outbound access to:
  - Internal DNS, NTP, OS package mirror, and container registry.
  - `ifflogin.fz-juelich.de`.
  - The configured Blablador endpoint and/or `api.openai.com`.
  - `github.com`, `gitlab.com`, and `iffgit.fz-juelich.de`.
  - `mcp.context7.com` and `api.tavily.com` only when enabled.
  - Docker Hub, GHCR, PyPI, and Python package CDN endpoints only if images are built on the VM. Prefer immutable images from an institutional registry.
- Obtain before go-live:
  - Public JüNA hostname and institutional TLS proxy configuration.
  - Stable SAML SP signing certificate/key and completed IdP registration.
  - Exact institutional proxy CIDRs.
  - LLM and optional Context7/Tavily credentials.
  - Random production PostgreSQL credentials.
- Configure the institutional proxy for WebSockets, requests/uploads of at least 100 MB, and long-running chat connections.
- Back up PostgreSQL, generated artifacts, SAML keys, and production configuration nightly. Repository clones and vector indices are reconstructable. Test restoration at least quarterly.

## Application and Deployment Changes

- Reuse one versioned JüNA image for three Compose services with different commands:
  - `api`: FastAPI only.
  - `ui`: Streamlit only.
  - `bootstrap`: one-shot repository synchronization and indexing.
- Add:
  - `gateway`: Caddy in HTTP-only mode. Route `/auth/*` to FastAPI and everything else, including `/_stcore/*`, to Streamlit.
  - `postgres`: internal-only persistent database.
- Install the sandbox worker on the host as the unprivileged `juena-sandbox`
  user. It communicates with JüNA only through `sandbox_jobs` in Postgres; it is
  not a Compose service and exposes no application-facing API.
- Make API startup depend on healthy PostgreSQL and successful bootstrap; make UI and gateway depend on healthy upstream services.
  - This keeps bootstrap all-or-nothing, which is a deliberate choice, not an
    oversight. `bootstrap_repositories` re-raises on the first repository
    failure and the entrypoint runs it under `set -e`, so one unreachable git
    remote out of the nine configured in `config/repositories.yaml` blocks
    startup — including general neutron-science chat, which needs no index.
    The agent already degrades gracefully on a missing index
    (`describe_bootstrap_problems`), so the strictness comes from the
    entrypoint, not the agent. Revisit if repository availability turns out to
    be the main source of downtime; until then, fail loudly rather than serve
    silently degraded retrieval.
- Persist PostgreSQL, repositories, indices, manifests, logs, and generated artifacts in named volumes. Keep sandbox workspaces on a separate host path owned only by the unprivileged sandbox account.
- Trust forwarded headers only from the institutional proxy CIDRs. Keep Streamlit CORS and XSRF protection enabled.
- Introduce `JUENA_PUBLIC_URL`; derive the SAML entity, ACS, SLS, login, logout, and UI URLs from it unless explicitly overridden. Compose supplies the non-user-configurable internal API URL `http://api:8080`.
- Remove the entire sidebar “Server” section: URL input, connected/disconnected indicator, and “Check” button.
- Remove `server_url` from Streamlit session state. On backend failure, show only a friendly service-unavailable message without internal hostnames.
- Use a session-scoped pooled HTTP client and close it cleanly instead of module-level HTTP calls.
- Retain exactly two immediate-send starter chips on empty threads:
  - **General neutron science** → “Give me a brief introduction to neutron science and the kinds of neutron-science questions you can help me answer.”
  - **Neutron software and codes** → “List all neutron-related software repositories currently indexed in JüNA, briefly explain what each one is for, and ask which repository I would like to explore.”
- Remove starter-only skill discovery code and `/skills` if nothing else consumes it; retain `SkillsMiddleware` for the agent.
- Correct the existing reviewed regressions:
  - Configure model retry with three retries and `on_failure="error"` so LangChain fallback receives exhausted failures.
  - Build the configured fallback independently of whichever provider first created the cached graph.
  - Bump the index schema/chunking fingerprint, force one rebuild, and persist exact chunk start/end offsets.
  - Remove the authenticated global agent-restart endpoint; provider/model remain request-scoped.
  - Update chat recency whenever a thread receives a message.
  - Treat zero configured repositories as repository research unavailable.
  - Remove dead/decorative middleware helpers and stale two-agent documentation without adding replacement abstractions.

## Python/Shell Sandbox and Plot Streaming

The current design is documented in [SANDBOX.md](SANDBOX.md). Deployment
obligations that remain in scope for this document:

- Install rootless Podman under an unprivileged `juena-sandbox` account.
- Install the host-side worker under that account. Only it reaches the rootless
  Podman socket; JüNA receives neither Docker nor Podman sockets.
- Give JüNA and the worker access to the same `sandbox_jobs` table and shared
  workspace root, with the ownership model from `deploy/SANDBOX.md`.
- Reserve approximately 140 GB for disposable sandbox workspaces on a host path
  owned only by `juena-sandbox`.
- Size for a maximum of four concurrent sandbox jobs.

## Verification and Acceptance

- Preserve the current passing baseline of 232 unit tests and four PostgreSQL integration tests.
- Add tests proving model fallback runs after retry exhaustion and is independent of graph creation order.
- Verify the chunk schema change causes exactly one rebuild and search results retain correct source offsets.
- Verify Compose exposes only the gateway port and uses service-name DNS, health-gated startup, non-root application processes, persistent volumes, and bounded log rotation.
- Test the complete institutional-proxy flow: HTTPS origin, forwarded scheme, SAML login/logout, secure cookie, Streamlit WebSocket, and browser refresh.
- Verify the chatbot contains no server address, connection control, or internal hostname.
- Verify that no application container holds a Docker or Podman socket, the
  worker exposes no network service, and `sandbox_jobs` is its only application
  control channel.
- Load-test 30 active chat sessions with four simultaneous sandbox jobs; confirm bounded memory, stable PostgreSQL pools, and predictable queueing.
- Perform a documented backup-and-restore rehearsal before production sign-off.

Sandbox-specific acceptance criteria—container isolation, approval flow,
artifact delivery, recovery, and capacity—are in
[SANDBOX.md](SANDBOX.md#production-enablement-gates).

Assumptions: the first release stays on one VM despite the weaker shared-kernel boundary; only authenticated FZJ users can request execution; the institutional proxy terminates TLS; SearXNG and Valkey are not deployed because current JüNA code does not consume them.
