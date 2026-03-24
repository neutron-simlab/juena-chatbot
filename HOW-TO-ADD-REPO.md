# How to Add a New Code Repository to `code_chat_agent`

This guide is for one specific job: making another codebase searchable by the
existing `code_chat_agent`.

You do not need to create a new agent or change `main.py`. You only need to:

1. Add the repo to `config/repositories.yaml`
2. Fully restart the application
3. Trigger `code_chat_agent` once so it clones and indexes the repo

## What Actually Happens

The lifecycle is:

1. `config/repositories.yaml` is loaded into `RepoManager`
2. `/repositories` shows the configured repo metadata after a full app restart
3. The first time `code_chat_agent` is created, it:
   - clones the repo into `REPO_CACHE_DIR`
   - builds a Chroma vector index in `VECTOR_INDEX_DIR`
   - exposes that repo to the search tools used by the agent

That means:

- editing the YAML file alone is not enough
- `/repositories` confirms the repo is registered
- a real `code_chat_agent` request is what triggers clone + indexing

## Step 1: Make Sure the Repo Is Reachable

The app uses `git clone` under the hood. The URL in the config must be
cloneable from the environment where the app runs.

Examples:

- public GitHub repo: `https://github.com/org/project.git`
- private repo over SSH: `git@github.com:org/project.git`
- local repo for local development: `file:///absolute/path/to/repo`

Notes:

- If you run with Docker, the container must have network access and any
  required SSH credentials.
- A `file://` path only works if that path exists inside the runtime
  environment. For Docker, that usually means mounting the path into the
  container first.

## Step 2: Add the Repo to `config/repositories.yaml`

Keep the top-level `repositories:` key. Add a new item under it.

Minimal example:

```yaml
repositories:
  - id: my-library
    source:
      url: https://github.com/my-org/my-library.git
      branch: main
```

More realistic example:

```yaml
repositories:
  - id: my-library
    name: My Library
    description: "Internal service for document ingestion and retrieval."
    source:
      url: https://github.com/my-org/my-library.git
      branch: main
    include_globs:
      - "**/*.py"
      - "**/*.md"
      - "**/*.yaml"
      - "**/*.yml"
      - "**/*.toml"
    exclude_globs:
      - "**/.git/**"
      - "**/__pycache__/**"
      - "**/.venv/**"
      - "**/node_modules/**"
      - "**/dist/**"
      - "**/build/**"
    max_file_bytes: 524288
    chunk_size: 1500
    chunk_overlap: 200
    docs_paths:
      - "README.md"
      - "docs/"
```

Important details:

- `id` must be unique. This becomes the repo identifier used by the agent.
- `source.url` is required.
- `source.branch` defaults to `main` if omitted.
- There is no `source.type` field in the current implementation. Do not add it.
- `name` defaults to `id` if omitted.

## Supported Fields

Each repository entry supports these fields:

| Field | Required | Default | Purpose |
|---|---|---|---|
| `id` | yes | none | Unique repo slug used by the agent and vector collection |
| `name` | no | same as `id` | Human-readable label |
| `description` | no | `""` | Short description shown in metadata |
| `source.url` | yes | none | Git URL used for cloning |
| `source.branch` | no | `main` | Branch to clone and track |
| `include_globs` | no | common text/code globs | Files eligible for indexing |
| `exclude_globs` | no | common cache/build globs | Files skipped during indexing |
| `max_file_bytes` | no | `524288` | Skip files larger than 512 KB |
| `chunk_size` | no | `1500` | Chunk size used for embedding |
| `chunk_overlap` | no | `200` | Overlap between chunks |
| `docs_paths` | no | `["README.md", "docs/"]` | Paths treated as documentation |

Default include globs:

```yaml
- "**/*.py"
- "**/*.md"
- "**/*.json"
- "**/*.yaml"
- "**/*.yml"
- "**/*.toml"
- "**/*.txt"
- "**/*.rst"
```

Default exclude globs:

```yaml
- "**/node_modules/**"
- "**/.git/**"
- "**/__pycache__/**"
- "**/.venv/**"
- "**/dist/**"
- "**/build/**"
- "**/*.egg-info/**"
```

## Step 3: Fully Restart the App

This part matters. A full restart is the safe way to make the new repo visible.

Local run:

```bash
uv run python main.py
```

Docker run:

```bash
docker compose up --build -d
```

Why `--build` is needed in Docker:

- the `Dockerfile` copies `config/` into the image at build time
- editing `config/repositories.yaml` on the host does not update the running
  container unless you rebuild the image

Do not rely on `/code_chat_agent/restart` for config reloads. That endpoint
restarts the agent graph, but it does not reliably reload the repository tool
context after `repositories.yaml` changes.

## Step 4: Verify the Repo Is Registered

First verify that the app loaded the repo config:

```bash
curl http://localhost:8080/repositories
```

You should see your new repo in the JSON response.

Also verify the agent exists:

```bash
curl http://localhost:8080/agents
```

You should see `code_chat_agent` in the `agents` list.

## Step 5: Trigger Clone and Index Build

Clone and indexing now happen during startup, before Streamlit and FastAPI come
up. That means the repo should already be ready once the container is healthy.

To watch bootstrap and indexing progress in Docker:

```bash
docker compose logs -f juena-chatbot
```

The logs include:

- bootstrap progress across repositories
- per-repository indexing progress as percentages

Example progress lines:

```text
Bootstrap progress: 33% (1/3 repositories completed)
Indexing progress for repo langgraph: 40% (120/300 files, 980 chunks)
```

## Choosing Good Settings

Use these rules of thumb:

- Small or medium Python repo: defaults are usually fine
- Monorepo: narrow `include_globs` to the relevant package or service
- Large generated trees: expand `exclude_globs`
- Docs-heavy repo: keep `docs_paths` accurate so `search_docs_local` stays useful

Examples:

- index only one package:
  `packages/core/**/*.ts`
- treat additional docs as documentation:
  `guides/`
- skip vendored code:
  `third_party/**`

## Adding Multiple Repositories

Just append more items under `repositories:`.

Example:

```yaml
repositories:
  - id: repo-a
    source:
      url: https://github.com/org/repo-a.git
      branch: main

  - id: repo-b
    source:
      url: https://github.com/org/repo-b.git
      branch: master
```

The agent will discover both through `list_repositories`.

## Updating an Existing Repo Later

There are two different cases:

1. You added a brand-new repo ID
2. You changed an already-indexed repo's branch, globs, or content and need a
   fresh semantic index

Case 1 is handled by the normal workflow above.

Case 2 may need a forced rebuild of the vector index, because existing Chroma
collections are persisted on disk. The simplest reset is:

Local:

```bash
rm -rf data/vector_index
uv run python main.py
```

Docker:

```bash
docker compose down
docker volume rm juena-chatbot-vector-index
docker compose up --build -d
```

That rebuilds the semantic index from scratch on the next `code_chat_agent`
request.

## Useful Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `REPO_CONFIG_PATH` | `config/repositories.yaml` | Override the repo config file location |
| `REPO_CACHE_DIR` | `data/repos` locally, `/data/repos` in Docker | Where repos are cloned |
| `VECTOR_INDEX_DIR` | `data/vector_index` locally, `/data/vector_index` in Docker | Where Chroma persists indices |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| New repo does not appear in `/repositories` | App did not restart with updated config | Restart locally, or rebuild Docker image with `docker compose up --build -d` |
| `git clone` fails | Bad URL or missing access | Check the URL and ensure the runtime has network/SSH credentials |
| Wrong branch error | `source.branch` is wrong | Check the repo's default branch and update the config |
| First question is slow | Clone + indexing is happening | Normal for the first `code_chat_agent` request |
| Docs search misses files | `docs_paths` does not match the repo layout | Add the correct paths and rebuild the index if needed |

## Short Version

If you just want the operator checklist:

1. Add a new item under `repositories:` in `config/repositories.yaml`
2. Set `id`, `source.url`, and the correct `source.branch`
3. Fully restart the app
4. Confirm the repo shows up in `GET /repositories`
5. Send one request to `code_chat_agent` so it clones and indexes the repo
