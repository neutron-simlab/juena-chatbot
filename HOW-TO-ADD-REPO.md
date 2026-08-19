# How to Add a New Code Repository to JüNA

This guide is for one specific job: making another codebase searchable by the
existing JüNA agent.

You do not need to create a new agent or change `main.py`. You only need to:

1. Add the repo to `config/repositories.yaml`
2. Rebuild/restart the application so startup bootstrap runs
3. Verify the repository appears in the authenticated discovery endpoint

## What Actually Happens

The lifecycle is:

1. `config/repositories.yaml` is loaded into `RepoManager`
2. `/repositories` shows the configured repo metadata after a full app restart
3. Startup clones the repo into `REPO_CACHE_DIR`, so the model can open a real
   file and quote real line numbers

**Cloning is not indexing.** This application no longer builds an index at all —
the vectors live in juena-rag, and an ingestion run there is what makes a
repository *searchable*. Adding an entry here without a matching entry in
juena-rag's own repository config gives you a repository JüNA can read file by
file but cannot search. `list_indexed_repositories` reports what is actually in
the store, which is the authoritative answer.

That means:

- editing the YAML file alone is not enough
- `/repositories` confirms the repo is registered here
- the repository must also be ingested into juena-rag before search finds it

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
| `include_globs` | no | common text/code globs | **Inert here** — see below |
| `exclude_globs` | no | common cache/build globs | **Inert here** — see below |
| `max_file_bytes` | no | `524288` | **Inert here** — see below |
| `chunk_size` | no | `1500` | **Inert here** — see below |
| `chunk_overlap` | no | `200` | **Inert here** — see below |
| `docs_paths` | no | `["README.md", "docs/"]` | **Inert here** — see below |

> **The last six fields no longer do anything in this application.** They are
> still parsed, so an existing config keeps loading, but nothing reads them:
> this repository clones, and juena-rag chunks and embeds. Editing
> `include_globs` here will not change what is searchable. The equivalent
> settings in juena-rag's own repository config are the ones that matter, and a
> change there takes effect on the next ingestion run.

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
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

Why `--build` is needed in Docker:

- the `Dockerfile` copies `config/` into the image at build time
- editing `config/repositories.yaml` on the host does not update the running
  container unless you rebuild the image

There is intentionally no shared agent-restart endpoint. A full process restart
reruns repository bootstrap and rebuilds the cached JüNA graph safely.

## Step 4: Verify the Repo Is Registered

First verify that the app loaded the repo config:

```bash
curl --cookie "juena_session=<opaque-session>" http://localhost:8080/repositories
```

You should see your new repo in the JSON response.

## Step 5: Trigger the Clone

Cloning happens during startup, before Streamlit and FastAPI come up, so the
working tree is on disk once the container is healthy. Nothing is indexed here.

To watch it in Docker:

```bash
docker compose logs -f juena-chatbot
```

The logs report each repository's revision as it is cloned or refreshed, then a
survey line naming any repository that is cloned but **not** in the store — that
is the one to take to juena-rag.

## Step 6: Ingest It Into juena-rag

Until this runs, the repository is readable file by file and invisible to
search. Ingestion lives in the juena-rag project, driven by its own repository
config; add a matching entry there and run its ingest for that source.

Confirm it worked by asking JüNA which repositories it can search — that calls
`list_indexed_repositories`, which reports the store's own catalogue and chunk
counts rather than what happens to be cloned on disk.

## Choosing Good Settings

These rules of thumb still hold, but **apply them in juena-rag's repository
config**, which is where chunking and file selection actually happen:

- Small or medium Python repo: defaults are usually fine
- Monorepo: narrow the include globs to the relevant package or service
- Large generated trees: widen the exclude globs
- Docs-heavy repo: keep the documentation paths accurate so `search_docs_local`
  stays useful

Setting any of these in `config/repositories.yaml` has no effect.

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

Case 2 is entirely juena-rag's business. There is no index on this side to
delete and no cache to clear: re-ingest the source there, which replaces its
chunks atomically, and this application sees the new content on its next search
with no restart. Restarting the chatbot refreshes the *clone*, which is what
`read_file` and `grep` see — a different thing from what search returns, and
worth keeping straight when the two disagree.

## Useful Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `REPO_CONFIG_PATH` | `config/repositories.yaml` | Override the repo config file location |
| `REPO_CACHE_DIR` | `data/repos` locally, `/data/repos` in Docker | Where repos are cloned |
| `JUENA_RAG_URL` | none — required | The juena-rag service that holds the vectors and answers every search |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| New repo does not appear in `/repositories` | App did not restart with updated config | Restart locally, or rebuild with `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d` |
| `git clone` fails | Bad URL or missing access | Check the URL and ensure the runtime has network/SSH credentials |
| Wrong branch error | `source.branch` is wrong | Check the repo's default branch and update the config |
| Service startup is slow | A large repository is cloning | Watch the bootstrap log and wait for the health check |
| Repo is listed but search finds nothing | It was never ingested into juena-rag | Add it there and run the ingest; `list_indexed_repositories` confirms |
| Search is stale but files look right | The clone refreshed; the store did not | Re-ingest in juena-rag — restarting this application will not do it |
| Docs search misses files | `docs_paths` does not match the repo layout | Fix it in **juena-rag's** config, not this one, then re-ingest |

## Short Version

If you just want the operator checklist:

1. Add a new item under `repositories:` in `config/repositories.yaml`
2. Set `id`, `source.url`, and the correct `source.branch`
3. Fully restart the app
4. Confirm the repo shows up in `GET /repositories`
5. Add the same repository to juena-rag's config and run its ingest — **without
   this the repository can be read but not searched**
6. Ask JüNA which repositories it can search; the new one should be listed with
   a non-zero chunk count
