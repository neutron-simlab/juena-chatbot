# Image build record

One entry per accepted `juena-chatbot` image, newest first.

## Why this file exists

`juena-core` is not fetched from a package index. It is the sibling directory
`../juena-core`, declared in `pyproject.toml` as:

```toml
[tool.uv.sources]
juena-core = { path = "../juena-core" }
```

`uv.lock` fixes the exact version of every dependency downloaded from the
internet, and it can verify those downloads by checksum. It cannot do that for a
directory on this machine: a local folder has no fixed checksum, because
anyone can edit it between one build and the next. So the lock file proves
nothing about which core source went into an image.

Two facts close that gap, and this file is where they are written down:

- **the core commit** the build copied, together with the fact that core's
  working tree had no uncommitted edits at the time. A dirty tree means the
  image contains code that exists nowhere in version control;
- **the image ID**, the checksum Docker computes over the finished image. Given
  the same two, anyone can tell whether a running container is the image this
  record describes.

This is deliberately lighter than a published package with a signature, and it
is enough for one person building on one machine. **It stops being enough the
day a second person builds an image**, because then "the sibling directory" is a
different directory. That is the point at which core moves to a git tag and both
sources switch together.

## How to add an entry

```bash
git -C ../juena-core status --porcelain   # must print nothing
git -C ../juena-core rev-parse HEAD       # the core commit
git rev-parse HEAD                        # this repository's commit
./juena build
docker image inspect juena-chatbot:latest --format '{{.Id}}'
shasum -a 256 uv.lock
```

---

## 2026-09-16 — the runtime proof, with the subagent execution fix

The image the four step-4 conversations were proved against. Same build shape as
the entry below; what changed is core.

| | |
|---|---|
| **Image ID** | `sha256:19434757dcfd3440d7e36e432a892c808819816d3273fda48082cc5347adca0b` |
| **juena-chatbot** | `0767d3913a57f6e92807338a6179f68cfe9d3dc3`, tree clean |
| **juena-core** | `a64c978a7d5cb03c3dfebeabddfb29ae7f10dbb3`, tree clean |
| **juena-rag** | `b7b72954477e781c1bda0828e24c4ecf0ffbccd0`, pinned by `uv.lock` |
| **Resolved set** | 160 packages; `uv.lock` sha256 `71b114dce71a2dfcb3234afe17078a8e30da6ee7020c276503188549511c6a7d` — unchanged from the previous entry |
| **Credentials used** | none |

Core carries one fix relative to the previous entry: sandbox `execute` was
raising `RuntimeError: Sandbox execution requires a graph run_id` for every
command, because the id it looked for does not exist inside a subagent — which
is the only place that tool is ever bound. See the plan's step-4 record.

The application source is identical between the two images; the commits after
the first entry changed `./juena`, `tests/` and `src/juena/research/runner.py`,
and only the last of those is copied in. Rebuilding from the clean commit above
reproduced this exact image ID.

---

## 2026-09-16 — the juena-core cutover, first parent-context build

The first image built after juena-chatbot stopped carrying the shared
infrastructure and started consuming `juena-core` (plan 02, step 3.5).

| | |
|---|---|
| **Image ID** | `sha256:352c43f42b5f8638523de87f5c73677dc2cd14cbc31ff6f94570ef6422f9fd10` |
| **juena-chatbot** | `2a9c1d785106b23855dceea53d9538fb757f1c7d`, tree clean |
| **juena-core** | `4eca16e99afba03023fadc22a8a9a37e2a43c9b4`, tree clean |
| **juena-rag** | `b7b72954477e781c1bda0828e24c4ecf0ffbccd0`, pinned by `uv.lock` |
| **Resolved set** | 160 packages; `uv.lock` sha256 `71b114dce71a2dfcb3234afe17078a8e30da6ee7020c276503188549511c6a7d` |
| **Base image** | `python:3.11-slim` |
| **Built by** | `./juena build`, which is `docker compose build` over the two development Compose files |

**Credentials used: none.** `juena-core` is copied from the build context, and
`juena-rag` is cloned anonymously over public HTTPS. Nothing in the build reads
a token, an SSH key or a registry login.

### What changed in this build, and why

The build context — the set of files Docker is allowed to read — moved from this
repository to its **parent directory**. A Dockerfile can only copy files from
inside its context, and `juena-core` sits *beside* this repository rather than
inside it, so with the old `context: .` there was no way to reach it.

Three consequences, all of them visible in the files:

1. **Every `COPY` in the `Dockerfile` now starts with a repository name** —
   `COPY juena-chatbot/src/ ./src/` rather than `COPY src/ ./src/`.
2. **Core lands at `/juena-core`, not under `/app`.** uv reads the path
   `../juena-core` relative to the project directory, which is `/app` inside the
   image; one level up from `/app` is the filesystem root. uv builds a wheel
   from that directory during `uv sync`; nothing imports it from there at
   runtime, and the installed copy under `/app/.venv` is what the application
   loads.
3. **`.dockerignore` is no longer consulted.** Docker looks for an ignore file
   named after the Dockerfile first, and only falls back to one at the context
   root — and the context root is now the parent directory, which has none. So
   `Dockerfile.dockerignore` is the only filter in effect. This is the part
   worth checking on any change: the parent directory holds about nine
   gigabytes across a dozen unrelated repositories, several carrying real
   credentials in their own `.env`. The ignore file therefore excludes
   everything and re-admits only `juena-chatbot/` and `juena-core/`.

### What was verified

```text
build context transferred            2.79 MB   (unfiltered parent: ~9 GB)
.env / *.key / *.pem in context      none
.git / .venv in context              none
prompt files in context              SUPERVISOR.md + both AGENT.md present
image: juena_core resolves to        /app/.venv/lib/python3.11/site-packages/juena_core
image: juena resolves to             /app/src/juena
image: /app/.env                     absent
image: /app/tests                    absent
rebuild from the clean commit        same image ID
Podman socket in resolved Compose    none, in either service
```

The context was measured by building a throwaway image that copied the whole
context and listed it, rather than by reading the ignore file and believing it.
An ignore rule that silently fails to match looks exactly like one that works.
