# JüNA Chatbot

<div align="center">
  <img src="app/assets/logo.png" alt="JüNA logo" width="180"/>
</div>

JüNA is a Docker-first multi-agent chatbot service with a FastAPI backend, a Streamlit chat UI, persistent conversation state, and repo-aware retrieval for code and documentation questions.

## What It Does

- Runs a web UI on port `9501` and an API on port `8080`
- Supports multiple agents behind one server
- Streams responses over Server-Sent Events
- Persists LangGraph checkpoints and chat history in SQLite
- Clones and indexes configured repositories for repo-aware Q&A
- Supports OpenAI and Blablador providers

## Quick Start

1. Create your environment file:

   ```bash
   cp env.example .env
   ```

2. Edit `.env` and configure at least one LLM provider.

3. Start the app:

   ```bash
   docker compose up --build -d
   ```

4. Open the services:
   - Streamlit UI: <http://localhost:9501>
   - FastAPI server: <http://localhost:8080>
   - OpenAPI docs: <http://localhost:8080/docs>

5. Optional verification:

   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/agents
   curl http://localhost:8080/repositories
   ```

Docker volumes persist logs, SQLite databases, cloned repositories, and vector indices across restarts.

## Available Agents

- `react_agent`
  Default general assistant registered by the server.

- `code_chat_agent`
  Repo-aware assistant for searching indexed repositories, reading files, and answering implementation questions with retrieval-first behavior.

If you call `/invoke` or `/stream` without an agent ID, the server uses `react_agent`. To target a specific agent, use `/{agent_id}/invoke` or `/{agent_id}/stream`.

## Repository Indexing

Repository metadata comes from [`config/repositories.yaml`](config/repositories.yaml).

The first time `code_chat_agent` is used, it will:

1. clone or refresh each configured repository
2. build a local Chroma vector index
3. expose repo search tools to the agent

That first request can be slower than later requests because indexing happens on demand.

For adding or tuning repositories, see [`HOW-TO-ADD-REPO.md`](HOW-TO-ADD-REPO.md).

Example request:

```bash
curl -X POST http://localhost:8080/code_chat_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"message":"List the available repositories and tell me what you can search."}'
```

## Useful Endpoints

- `GET /health` checks whether the service is up
- `GET /agents` lists registered agent IDs and the default agent
- `GET /repositories` lists configured repository metadata
- `POST /invoke` and `POST /stream` talk to the default agent
- `POST /{agent_id}/invoke` and `POST /{agent_id}/stream` talk to a specific agent

## Notes

- `.env` is loaded from the repo root by default
- chat history and checkpoints are stored in SQLite
- repo cache and vector index paths are configurable with environment variables

## License

MIT
