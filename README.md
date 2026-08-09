# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA is a Docker-first chatbot stack with a FastAPI backend, a Streamlit UI, one authenticated LangGraph agent, Postgres persistence, and repository-aware research.

## Overview

- FastAPI API on `8080` and Streamlit UI on `9501`
- One JüNA agent for neutron science, software, and general questions
- SSE streaming for chat responses
- FZJ SAML2 login with user-owned conversations
- Postgres persistence for identity, sessions, chat history, checkpoints, and agent files
- Specialist-owned, optional Agent Skills with progressive disclosure
- Per-user Deep Agents memory stored as `AGENTS.md`
- Repository bootstrap and vector indexing for code-aware retrieval
- OpenAI and Blablador model support
- Optional Tavily web search

## Included Agent

`juena` is the only advertised agent. It is a strict supervisor over two
first-class specialists. `software-specialist` handles software, repositories,
and upstream documentation; `neutron-research-specialist` handles neutron
science and broader factual research in JueNA's domain. Both accept staged text
attachments through read-only `/inputs` access. Missing repository indexing,
Context7, or Tavily degrades only the corresponding capability and does not
remove either specialist.

An empty thread normally shows two immediate-send chips:

- **General neutron science**
- **Neutron software and codes** — asks JüNA to list and explain every indexed
  repository before asking which one the user wants to explore.

The development Compose stack also enables **Plot with Python**. It asks the
software specialist to generate a script, pauses for your approval, runs it in
an isolated container, and displays the resulting plot.

## Quick Start

1. Copy the environment template:

   ```bash
   cp env.example .env
   ```

2. Edit `.env` and configure at least one chat model provider. For production,
   also replace every development Postgres/SAML URL and mount the registered SP
   certificate and key.

   `POSTGRES_PASSWORD` is required and has no default: Compose refuses to start
   without it rather than handing the database an empty password. If you are
   upgrading an `.env` created before it was added, copy the `POSTGRES_*` block
   out of `env.example`.

3. Optional: update [`config/repositories.yaml`](config/repositories.yaml) to change the repositories available to JüNA.

4. Install and initialize rootless Podman once on macOS:

   ```bash
   brew install podman
   podman machine init --memory 8192
   ```

5. Start the complete development stack with the normal shortcut:

   ```bash
   ./juena up
   ```

   This starts the Podman VM when needed, builds the scientific image, starts
   the host broker, and then starts PostgreSQL and JüNA through Docker Compose.
   The fixed sandbox image currently includes `jscatter==1.9.0.1`; its pinned
   repository-software manifest is
   `src/juena/sandbox/repository-requirements.txt`.
   After running `./juena install` once, use `juena up` from any directory.

6. Watch bootstrap and startup logs:

   ```bash
   ./juena logs
   ```

7. Register the metadata URL with the FZJ IdP when deploying a real SP, then open:
   - Streamlit UI: <http://localhost:9501>
   - FastAPI server: <http://localhost:8080>
   - SP metadata: <http://localhost:8080/auth/metadata>

8. Optional verification:

   ```bash
   curl http://localhost:8080/health
   # Authenticated endpoints require the juena_session cookie.
   ```

The development stack is detached, so bootstrap progress will not appear
inline. Use `./juena logs` to watch JüNA and see the recent host-broker log. Use
`./juena sandbox-stop` to stop those two services while keeping PostgreSQL and
its data available.

Development explicitly merges
[`docker-compose.dev.yml`](docker-compose.dev.yml) after the base file. It
supplies localhost development URLs, publishes Postgres for host-side
integration tests, and points the app at the host rootless-Podman broker.
Development and production therefore use the same broker process and Podman
execution path; no application container receives a container-engine socket.
Because the filename is not an automatic override, production cannot load these
development settings accidentally. Production uses only the base file:

```bash
docker compose -f docker-compose.yml \
  --env-file /etc/juena/juena.env up -d
```

The base configuration does not publish PostgreSQL and does not supply
localhost SAML URLs or an insecure cookie override. It requires explicit
production secrets and public HTTPS settings. The future gateway and
split-service stack remain specified in
[`DEPLOYMENT-PLAN.md`](DEPLOYMENT-PLAN.md); the code-execution sandbox and plot
streaming in [`SANDBOX-PLAN.md`](SANDBOX-PLAN.md); the implemented supervisor,
specialist, and skills architecture in
[`PROMPT-AND-SKILLS-PLAN.md`](PROMPT-AND-SKILLS-PLAN.md).

## Startup and Bootstrap

Bootstrap runs before the app starts serving requests.

- The Docker entrypoint runs `python -m juena.indexing.bootstrap` before starting Streamlit or FastAPI.
- `main.py` runs the same bootstrap path when started directly, unless `JUENA_BOOTSTRAP_DONE=1` is already set.
- Bootstrap fails fast: if any configured repository cannot sync or index, startup stops.

For each repository, bootstrap:

1. clones or refreshes the repo
2. reads the current revision
3. checks whether the index is stale
4. selectively reindexes changed files when possible
5. falls back to a full rebuild when the index is missing, empty, config/embedding settings changed, or incremental sync is unsafe

Selective reindexing uses `git diff` from the indexed revision to the current revision:

- delete chunks for removed or renamed-out files
- re-embed only added, changed, and renamed-in files
- update metadata only when no indexed files changed
- fall back to manifest diffing or a full rebuild when needed

Logs show overall repository progress and per-repository file progress.

You can run bootstrap manually:

```bash
uv run python -m juena.indexing.bootstrap
```

## Repository Configuration

JüNA only knows about repositories listed in [`config/repositories.yaml`](config/repositories.yaml).

Each repository entry controls:

- the repository id and display name
- clone URL and branch
- which files are indexed
- which paths are treated as documentation
- chunk size and overlap for embedding

The authenticated API discovery endpoint:

```bash
curl --cookie "juena_session=<opaque-session>" http://localhost:8080/repositories
```

returns every configured repository. The **Neutron software and codes** starter
asks the agent to explain this complete set before inviting the user to choose
one.

## Authentication and ownership

The Streamlit UI does not generate or trust browser-session user IDs. It reads
the `juena_session` cookie, verifies it through `/auth/me`, and forwards the
opaque cookie to FastAPI. FastAPI derives one internal UUID from the persistent
NameID issued by the configured FZJ IdP and verifies thread ownership for every
chat, invocation, resume, rename, and delete operation.

The cookie token is stored only as a SHA-256 hash in Postgres. Sessions last
eight hours and are `HttpOnly`, `SameSite=Lax`, and `Secure` in production.
Authenticated users can use **Sign out** in the Streamlit sidebar. The browser
posts directly to `/auth/logout`, allowing FastAPI to revoke the server-side
session, clear the HttpOnly cookie, and either continue IdP logout or return to
the UI when SAML single logout is unavailable.

Three origins have distinct roles during the compatibility release:

| Setting | Purpose | Development default |
| --- | --- | --- |
| `JUENA_API_INTERNAL_URL` | Private Streamlit-to-FastAPI calls; never shown to the browser. | `http://localhost:8080` |
| `JUENA_PUBLIC_URL` | Browser-facing origin used for login links. Production requires explicit HTTPS. | `http://localhost:8080` |
| `UI_PUBLIC_URL` | Post-login UI destination retained for compatibility. In production it defaults to `JUENA_PUBLIC_URL`. | `http://localhost:9501` |

Production rejects missing or non-HTTPS public/SAML URLs and rejects
`SESSION_COOKIE_SECURE=false`. Streamlit keeps the internal URL in server-side
configuration; the chatbot has no server URL, connection indicator, or Check
control. Streamlit's CORS and XSRF protections retain their secure defaults.
Ordinary Streamlit-to-FastAPI requests use `TIMEOUT_SECONDS` (60 seconds by
default). Agent response streams leave the read timeout open while retaining
that finite limit for connection setup, writes, and connection-pool waits.

SAML endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /auth/login` | Start SP-initiated login (signed AuthnRequest). |
| `POST /auth/acs` | AssertionConsumerService; validates the assertion and sets the cookie. |
| `GET /auth/metadata` | SP metadata to register with the IdP. |
| `POST /auth/logout` | **Application** logout for the signed-in UI session. |
| `GET|POST /auth/sls` | **SAML** SingleLogoutService. Register this as `SAML_SP_SLS_URL`. |

`/auth/logout` and `/auth/sls` are deliberately separate: the IdP reaches
`/auth/sls` without our cookie — both to answer our LogoutRequest and to start
logout itself — and is authorized by SAML signature validation, whereas
`/auth/logout` requires an authenticated session. Pointing `SAML_SP_SLS_URL` at
`/auth/logout` will make IdP-initiated logout fail with 401.

## Specialists, skills, and memory

JüNA uses a strict supervisor with two explicitly registered specialists:
`software-specialist` owns repository and upstream-library research, while
`neutron-research-specialist` owns scientific and external factual research.
The supervisor owns delegation, memory, reconciliation, and the final answer;
it has no repository, Context7, or Tavily tools itself.

Each specialist package contains its own plain `AGENT.md`, model choice, and tool
builder. Optional skills belong under that package as
`skills/<name>/SKILL.md`. `SkillsMiddleware` is enabled for that specialist only
when at least one valid skill exists, so an empty skill library adds no prompt
overhead. New specialists are added explicitly rather than discovered through
filesystem imports.

The supervisor uses Deep Agents' native `MemoryMiddleware` with
`/memories/AGENTS.md`. That file is stored in the LangGraph Postgres Store under
the `("memories", <user-uuid>)` namespace, so it persists across threads without
being shared across users. Specialists receive only relevant preferences that
the supervisor includes in a delegated task, never the complete memory file.

The default Deep Agents memory prompt instructs the model to save anything it
judges useful. JueNA replaces it (`JUENA_MEMORY_SYSTEM_PROMPT` in
[`backends.py`](src/juena/agents/backends.py)) so memories are written **only**
when the user explicitly asks, never inferred, and never used for credentials or
secrets.

For a step-by-step guide to adding repositories, see [`HOW-TO-ADD-REPO.md`](HOW-TO-ADD-REPO.md).

## Embeddings and Providers

At least one chat provider must be configured in `.env`.

- Select the primary with `DEFAULT_PROVIDER=openai|blablador`.
- Set provider-specific defaults with `OPENAI_DEFAULT_MODEL` and
  `BLABLADOR_DEFAULT_MODEL`. The legacy `DEFAULT_MODEL` override remains
  accepted for one compatibility release.
- Set `FALLBACK_PROVIDER` to another configured provider when model failures
  should fall through after three retries.
- OpenAI currently offers `gpt-4o-mini` in the built-in picker. The deprecated
  GPT-4o, GPT-4 Turbo, and GPT-3.5 Turbo entries were removed; deployments can
  further constrain the picker with `OPENAI_AVAILABLE_MODELS`.
- Repository indexing can use Blablador's OpenAI-compatible embedding endpoint via:
  - `BLABLADOR_API_KEY`
  - `BLABLADOR_BASE_URL`
  - `BLABLADOR_EMBEDDING_MODEL`

If the Blablador embedding settings are missing, indexing falls back to Chroma's default embedding function.

Relevant settings are documented in [`env.example`](env.example).

## Context7 MCP

JüNA can optionally load Context7 tools for external dependency
documentation and code examples.

- Context7 is separate from local repository indexing.
- Juena uses one shared Context7 MCP server connection, not one server per
  library.
- Context7 is intended for popular upstream libraries/frameworks, while
  configured repositories still come from [`config/repositories.yaml`](config/repositories.yaml)
  and are cited from `/repos/...`.
- If `CONTEXT7_API_KEY` is not set, JüNA keeps working with only
  local repository tools.

Optional environment variables:

- `CONTEXT7_API_KEY`
- `CONTEXT7_MCP_URL` (default: `https://mcp.context7.com/mcp`)
- `CONTEXT7_TIMEOUT_SECONDS` (default: `30`)

## Tavily Web Search

JüNA can optionally give Tavily to the neutron-research specialist for current
or externally verifiable questions. The supervisor never receives the tool;
software and repository research remains limited to local tools plus Context7.

Optional environment variables:

- `TAVILY_API_KEY`

## Data and Persistence

The stack persists state across restarts.

- users, sessions, conversation ownership/titles: Postgres application tables
- conversation **messages**: the LangGraph checkpointer tables, which are the
  single source of truth. There is deliberately no mirror table — `GET
  /chats/{thread_id}` reads history back from the checkpointer, so the UI and
  the agent can never show different histories. Pass `include_messages=false`
  for metadata-only reads.
- per-user `AGENTS.md`: the LangGraph Postgres Store
- cloned repositories: `data/repos`
- vector indices: `data/vector_index`

Two connection pools serve one database: SQLAlchemy for the application tables,
and a single psycopg pool (`DATABASE_POOL_MAX_SIZE`, default 20) shared by the
checkpointer and the Store. `DATABASE_URL` may use `postgres://`,
`postgresql://`, or `postgresql+psycopg://`; all three are normalized.

The application creates its schema directly from the SQLAlchemy models on a
clean database; Alembic migrations are intentionally not used at this stage.
Existing anonymous SQLite files are mounted only as a read-only archive and are
never assigned to authenticated users or imported.

Run `uv run pytest -q` for the unit suite. The Postgres-backed ownership,
checkpoint, and Store tests need a live development database:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres
JUENA_TEST_DATABASE_URL=postgresql://juena:juena-development-only@localhost:5432/juena \
  uv run pytest tests/test_postgres_integration.py -q
```

For a deliberate clean development reset, stop the stack and remove its
`juena-chatbot-postgres` volume before restarting. This destroys the development
Postgres data and should never be used for production upgrades.

## Notes

- `.env` is loaded from the repository root by default.
- In Docker, the entrypoint also supports `/etc/juena/.env`.
- If you intentionally change the embedding model for code-chat indexing, rebuild the vector index before relying on old collections.

## License

MIT
