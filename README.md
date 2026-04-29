# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA is a Docker-first chatbot stack with a FastAPI backend, a Streamlit UI, LangGraph agents, persistent SQLite state, and a repository-aware code-chat workflow.

## Overview

- FastAPI API on `8080` and Streamlit UI on `9501`
- Two built-in agents behind one server
- SSE streaming for chat responses
- SQLite persistence for LangGraph checkpoints and chat history
- Repository bootstrap and vector indexing for code-aware retrieval
- OpenAI and Blablador model support
- Optional Tavily web search

## Included Agents

- `react_agent`
  The default general-purpose assistant. Bare `/invoke` and `/stream`
  requests go here unless you specify another agent id. When `TAVILY_API_KEY`
  is configured, it can also use Tavily web search for current or external
  facts.

- `code_chat_agent`
  A repository-analysis workflow for indexed code and docs. It combines:
  - a top-level coordinator that plans with todos and delegates repository investigation
  - a dedicated repository expert subagent for local code search, file reading, and staged inputs
  - read-only access to cloned repositories under `/repos/<repo_id>/`
  - hybrid retrieval for code and documentation
  - semantic search over persistent vector indices
  - optional Context7 MCP access for external library/framework docs and code examples
  - optional coordinator-level Tavily web search for non-repository or current-event questions

On a fresh UI chat, the welcome message introduces both agents and lists the repositories available to `code_chat_agent`.

## Quick Start

1. Copy the environment template:

   ```bash
   cp env.example .env
   ```

2. Edit `.env` and configure at least one chat model provider.

3. Optional: update [`config/repositories.yaml`](config/repositories.yaml) if you want `code_chat_agent` to index different repositories.

4. Start the stack:

   ```bash
   docker compose up --build -d
   ```

5. Watch bootstrap and startup logs:

   ```bash
   docker compose logs -f juena-chatbot
   ```

6. Open the services:
   - Streamlit UI: <http://localhost:9501>
   - FastAPI server: <http://localhost:8080>
   - OpenAPI docs: <http://localhost:8080/docs>

7. Optional verification:

   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/agents
   curl http://localhost:8080/repositories
   ```

`docker compose up --build -d` is detached, so bootstrap progress will not appear inline in the compose output. Use `docker compose logs -f juena-chatbot` to watch indexing progress.

## Startup and Bootstrap

Bootstrap runs before the app starts serving requests.

- The Docker entrypoint runs `python -m juena.retrieval.bootstrap` before starting Streamlit or FastAPI.
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
uv run python -m juena.retrieval.bootstrap
```

## Repository Configuration

`code_chat_agent` only knows about repositories listed in [`config/repositories.yaml`](config/repositories.yaml).

Each repository entry controls:

- the repository id and display name
- clone URL and branch
- which files are indexed
- which paths are treated as documentation
- chunk size and overlap for embedding

The current API discovery endpoint:

```bash
curl http://localhost:8080/repositories
```

returns the configured repository metadata that the UI also uses for the code-chat welcome.

For a step-by-step guide to adding repositories, see [`HOW-TO-ADD-REPO.md`](HOW-TO-ADD-REPO.md).

## Embeddings and Providers

At least one chat provider must be configured in `.env`.

- OpenAI can be used for chat models.
- Blablador can be used for chat models.
- Repository indexing can use Blablador's OpenAI-compatible embedding endpoint via:
  - `BLABLADOR_API_KEY`
  - `BLABLADOR_BASE_URL`
  - `BLABLADOR_EMBEDDING_MODEL`

If the Blablador embedding settings are missing, indexing falls back to Chroma's default embedding function.

Relevant settings are documented in [`env.example`](env.example).

## Context7 MCP

`code_chat_agent` can optionally load Context7 tools for external dependency
documentation and code examples.

- Context7 is separate from local repository indexing.
- Juena uses one shared Context7 MCP server connection, not one server per
  library.
- Context7 is intended for popular upstream libraries/frameworks, while
  configured repositories still come from [`config/repositories.yaml`](config/repositories.yaml)
  and are cited from `/repos/...`.
- If `CONTEXT7_API_KEY` is not set, `code_chat_agent` keeps working with only
  local repository tools.

Optional environment variables:

- `CONTEXT7_API_KEY`
- `CONTEXT7_MCP_URL` (default: `https://mcp.context7.com/mcp`)
- `CONTEXT7_TIMEOUT_SECONDS` (default: `30`)

## Tavily Web Search

`react_agent` can optionally use Tavily for web search. `code_chat_agent`
can also use Tavily at the coordinator level for non-repository or current web
questions, while its repository expert subagent stays limited to local repo
tools plus Context7.

Optional environment variables:

- `TAVILY_API_KEY`

## Data and Persistence

The stack persists state across restarts.

- chat history: `data/db/chats.sqlite`
- LangGraph checkpoints: `data/db/checkpoints.sqlite`
- cloned repositories: `data/repos`
- vector indices: `data/vector_index`

Docker volumes preserve those paths by default.

## Notes

- `.env` is loaded from the repository root by default.
- In Docker, the entrypoint also supports `/etc/juena/.env`.
- If you intentionally change the embedding model for code-chat indexing, rebuild the vector index before relying on old collections.

## License

MIT
