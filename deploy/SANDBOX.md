# Sandbox broker deployment

The sandbox broker is a second service. It runs model-generated commands in
rootless Podman containers; the main JüNA container must never receive a Podman
or Docker socket.

Development and production use the same broker code and rootless-Podman command.
On macOS, Homebrew Podman runs the containers in its rootless Linux VM. On
production Linux, systemd runs the broker under the dedicated `juena-sandbox`
account. The JüNA application reaches the host broker over HTTP and receives no
container-engine socket.

## Local development

Install Podman, initialize its rootless VM once, and start JüNA:

```bash
brew install podman
podman machine init --memory 8192
juena up
```

`juena up` starts the Podman VM when needed, builds the fixed scientific image,
starts the broker
as an unprivileged host process, and then starts the Compose application stack.
The resulting topology has three pieces:

1. `postgres` stores users, chats, and checkpoints.
2. `juena-chatbot` runs the API and Streamlit UI and calls the host broker.
3. The host broker asks rootless Podman to run the fixed scientific image.

The generated-code container always uses `--network=none`; it is never attached
to the application or database network. Inputs and outputs cross the boundary
only through the broker-managed workspace. The local broker log and PID live in
the ignored `.sandbox/` directory.

The fixed image includes the general scientific Python stack plus the packages
listed in `src/juena/sandbox/repository-requirements.txt`. The first
repository-backed package is `jscatter==1.9.0.1`. Add future repository software
to that pinned manifest and rebuild the image; commands cannot install packages
or access package indexes at runtime.

Homebrew Podman mounts `/Users` into its VM, so the launcher keeps the local
workspace under the repository's `.sandbox/` directory. The same absolute path
is visible to the Podman VM and the native broker.

Use `juena logs` to see application logs and the latest broker output. Use
`juena sandbox-stop` to stop the app and broker while leaving PostgreSQL
running.

## Host preparation

These commands are examples for a Linux administrator. Package and service
commands differ between distributions.

1. Install rootless Podman and create a dedicated `juena-sandbox` system user.
2. Configure subordinate UID/GID ranges and enable lingering for that user so
   rootless Podman can run as a service.
3. Put this JüNA checkout and its virtual environment under `/opt/juena`.
4. Build the fixed scientific image as that user:

   ```bash
   podman build -t juena-sandbox:latest -f deploy/sandbox/Dockerfile .
   ```

5. Copy `deploy/sandbox-broker.env.example` to
   `/etc/juena/sandbox-broker.env`, generate a long random token, and make the
   file readable only by the broker administrator and service account.
6. Put `deploy/juena-sandbox-broker.service` in `/etc/systemd/system/`, then
   enable and start it.
7. Restrict TCP port `8090` with the host firewall so only the JüNA application
   host/network can reach it. The token is authentication, not a reason to
   expose the broker publicly.

The workspace root must be on a filesystem with per-directory/project quotas.
Set a hard 5 GB quota for every workspace directory. The broker also checks
size before and after jobs, but those checks cannot replace a filesystem quota
while an untrusted process is actively writing.

## JüNA configuration

Set the same `SANDBOX_BROKER_TOKEN` in JüNA, set `SANDBOX_ENABLED=true`, and set
`SANDBOX_BROKER_URL` to the broker's private address. For the provided Compose
deployment on the same Linux host, the default URL is
`http://host.docker.internal:8090`.

After restart, confirm the JüNA health endpoint and broker logs are clean. The
empty-chat UI will then show the `Plot with Python` starter. Every command still
requires a user click before execution.

## Threat model

The trust question is *not* "is the user malicious." It is: **model-authored code
runs on shared institutional infrastructure, and the human approving it owns
none of that infrastructure.**

With a local coding assistant, the operator usually owns the machine, files, and
consequences. In JüNA the user, the infrastructure owner, and the possible blast
radius are different parties, so user consent cannot transfer the institution's
risk or serve as the security boundary. Containment must.

Threats, in priority order:

1. **Prompt injection into executable code.** Indexed repository content and
   uploaded files are untrusted; either can steer the model toward code it would
   not otherwise write. Assume this succeeds — the container is what makes it
   harmless.
2. **Cross-tenant data access.** One user's uploads, memories, or artifacts
   reachable from another user's execution.
3. **Host and infrastructure escape.** Container breakout, socket access,
   credential or SAML material exposure.
4. **Resource exhaustion.** One thread starving the other chat sessions.
5. **Egress.** Exfiltration of uploaded scientific data.

Explicitly accepted: the model may produce *incorrect* scientific results. No
sandbox or approval gate addresses that — it is a UI and prompting concern, and
outputs must be presented as agent-generated and unverified.

## Before enabling in production

Keep `SANDBOX_ENABLED=false` in production until every item below passes. A
working development Podman VM and green unit tests are *not* sufficient evidence
to enable untrusted code execution on shared infrastructure — a Podman VM proves
nothing about target-host filesystem quotas, systemd identity, firewalling, Linux
host mounts, or production load behaviour.

1. Install the broker under a dedicated unprivileged account on the target Linux
   host, following the host preparation section above.
2. Build and inspect the image with rootless Podman on that host.
3. Configure a private/firewalled broker address and the shared random token.
4. Configure and prove a hard 5 GB filesystem/project quota per workspace. This
   quota is **required**, not advisory.
5. Decide whether to enable and test gVisor (`runsc`).
6. Repeat the complete `Plot with Python` agent/UI flow through the Podman path,
   including approval, inline plot, downloads, chat reload, and thread deletion.
7. Prove the job cannot reach the network, host filesystem, application data,
   database, credentials, SAML material, or container sockets.
8. Test approve, edit, reject, timeout, broker outage, replay, cross-user access,
   malformed files, quota exhaustion, and cleanup against the deployed broker.
9. Run the planned load test with 30 active chat sessions and four simultaneous
   sandbox jobs.
