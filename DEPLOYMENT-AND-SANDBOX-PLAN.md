# JüNA Production Deployment and Sandbox Plan

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
- Do not publish PostgreSQL, FastAPI, Streamlit, Podman, or sandbox-broker ports externally.
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
  - `sandbox-broker`: internal API for controlled code execution.
- Make API startup depend on healthy PostgreSQL and successful bootstrap; make UI and gateway depend on healthy upstream services.
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

- Integrate execution through Deep Agents’ `SandboxBackendProtocol`; never use `LocalShellBackend` in production.
- Run a dedicated rootless Podman service under an unprivileged `juena-sandbox` account. Only the broker may access its socket; FastAPI and the agent containers must not receive either Docker or Podman sockets.
- Create a per-thread workspace and start an ephemeral container for each approved execution.
- Initial scientific image contains Python 3.11, NumPy, SciPy, pandas, matplotlib, seaborn, SymPy, scikit-learn, Pillow, and basic shell utilities. Runtime package installation is disabled.
- Apply these default limits:
  - Four concurrent jobs; excess jobs queue.
  - 2 vCPU, 4 GB RAM, 256 PIDs, 1 GB temporary filesystem per job.
  - Ten-minute hard execution timeout.
  - 5 GB per-thread workspace and 24-hour idle expiry.
  - Read-only root filesystem, non-root user, all Linux capabilities dropped, `no-new-privileges`, default seccomp confinement, and no network.
  - Maximum 100 MB uploaded input per turn.
  - Truncate excessive stdout/stderr before returning it to the model.
- Copy accepted user files into the thread workspace as read-only inputs; never mount application repositories, database data, credentials, or SAML material into a sandbox.
- Add LangChain `HumanInTheLoopMiddleware` for the `execute` tool with approve, edit, and reject decisions. The UI may edit command arguments but may not change the tool name or resource policy.
- Add an authenticated resume SSE endpoint that resumes the existing LangGraph checkpoint using the interrupt ID and `Command(resume=...)`. Reject stale, replayed, cross-thread, or cross-user decisions.
- Stream these new structured events:
  - `approval_required`: interrupt ID, command, description, limits, and allowed decisions.
  - `sandbox_status`: queued, running, completed, timed out, or rejected.
  - `artifact`: artifact ID, filename, MIME type, size, and base64 payload.
- Generated plots go under `/workspace/outputs`. After execution, validate and import PNG images into persistent artifact storage.
- Limit plots to four images per execution, 5 MB per image, and safe pixel dimensions. Reject malformed data and active formats such as SVG.
- Use LangChain custom streaming/`ToolMessage.artifact` so raw base64 is never placed in model context. Streamlit decodes validated base64 and renders the bytes with `st.image`.
- Persist artifact metadata by authenticated user and thread. Provide authenticated artifact retrieval for chat reloads, and delete associated files when the thread is deleted.
- Record an audit entry containing user, thread, interrupt, original and edited commands, decision, sandbox job ID, timestamps, exit status, and generated artifact references.

## Verification and Acceptance

- Preserve the current passing baseline of 218 unit tests and four PostgreSQL integration tests.
- Add tests proving model fallback runs after retry exhaustion and is independent of graph creation order.
- Verify the chunk schema change causes exactly one rebuild and search results retain correct source offsets.
- Verify Compose exposes only the gateway port and uses service-name DNS, health-gated startup, non-root application processes, persistent volumes, and bounded log rotation.
- Test the complete institutional-proxy flow: HTTPS origin, forwarded scheme, SAML login/logout, secure cookie, Streamlit WebSocket, and browser refresh.
- Verify the chatbot contains no server address, connection control, or internal hostname.
- Test approve, edited approval, rejection, timeout, replay prevention, ownership enforcement, queue saturation, workspace cleanup, and thread deletion.
- Demonstrate that sandbox jobs cannot reach the network, application volumes, environment secrets, container sockets, PostgreSQL, or the host filesystem.
- Run an end-to-end matplotlib test: approved Python creates a PNG, the API streams validated base64, Streamlit renders it, and it remains available after reloading the chat.
- Load-test 30 active chat sessions with four simultaneous sandbox jobs; confirm bounded memory, stable PostgreSQL pools, and predictable queueing.
- Perform a documented backup-and-restore rehearsal before production sign-off.

Assumptions: the first release stays on one VM despite the weaker shared-kernel boundary; only authenticated FZJ users can request execution; sandboxes have no egress or runtime package installation; the institutional proxy terminates TLS; SearXNG and Valkey are not deployed because current JüNA code does not consume them.
