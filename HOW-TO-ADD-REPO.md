# How to Add a New Code Repository to JüNA

This guide is for one specific job: making another codebase searchable by the
existing JüNA agent.

You do not need to create a new agent or change `main.py`. You only need to:

1. Add the repo to `config/repositories.yaml`
2. Rebuild/restart the application so startup bootstrap runs
3. Ingest the repository into juena-rag
4. Verify registration and a nonzero indexed chunk count

Examples below use the local launcher. On the Linux development server, use
`sudo ./juena prod ...` so commands read `/etc/juena/juena.env`. The checkout's
`.env` is for local development. Both shared and standalone gateways use the
same repository workflow.

## What Actually Happens

The lifecycle is:

1. `config/repositories.yaml` is loaded into `RepoManager`
2. `/repositories` shows the configured repo metadata after a full app restart
3. Startup clones the repo into `REPO_CACHE_DIR`, so the model can open a real
   file and quote real line numbers

**Cloning is not indexing.** This application no longer builds an index at all —
the vectors live in juena-rag, and an ingestion run is what makes a repository
*searchable*. That run reads this same file: `./juena repos-ingest` hands
`config/repositories.yaml` to juena-rag, which keeps no repository list of its
own. Until it runs for a new entry, JüNA can read that repository file by file
but cannot search it. `list_indexed_repositories` reports what is actually in the
store, which is the authoritative answer.

That means:

- editing the YAML file alone is not enough
- `/repositories` confirms the repo is registered here
- `./juena repos-ingest <id>` must run before search finds it

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
    chunk_size: 3000
    chunk_overlap: 400
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
    chunk_size: 3000
    chunk_overlap: 400
    docs_paths:
      - "README.md"
      - "docs/"
```

Important details:

- `id` must be unique. This becomes the repo identifier used by the agent and
  the source id in juena-rag.
- `source.url` is required.
- `chunk_size` and `chunk_overlap` are required: the ingest refuses an entry
  without them. 3000 and 400 are what every repository uses today.
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
| `include_globs` | no | `**/*.py`, `**/*.md`, `**/*.rst`, `**/*.txt` | Which files are indexed |
| `exclude_globs` | no | common cache/build globs | Which files are skipped |
| `max_file_bytes` | no | `524288` | Larger files are skipped |
| `chunk_size` | **yes** | none | Characters per chunk |
| `chunk_overlap` | **yes** | none | Overlap between chunks |
| `docs_paths` | no | none | Files and folders counted as documentation, for `search_docs_local` |

> **The last six fields are read by the ingest, not by this application.** This
> application only clones. `./juena repos-ingest` passes the file to
> `juena-rag ingest-repo`, which selects files and splits them by these fields,
> so the defaults above are juena-rag's. A change takes effect at the next ingest
> of that repository. A repository whose commit has not moved is skipped, so after
> a settings-only change run `./juena repos-ingest <id> --force`.

## Step 3: Fully Restart the App

This part matters. A full restart is the safe way to make the new repo visible.

Local run:

```bash
uv run python main.py
```

Local Docker run:

```bash
./juena up
```

Linux server:

```bash
sudo ./juena prod up
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

This is a local API example. The server does not publish port 8080; use
`sudo ./juena prod doctor` and check `rag.configured`, or ask the signed-in
agent to list configured repositories. You should see the new ID. A registered
ID alone does not establish that it has been indexed.

## Step 5: Trigger the Clone

Cloning happens during startup, before Streamlit and FastAPI come up, so the
working tree is on disk once the container is healthy. Nothing is indexed here.

To watch it in Docker:

```bash
./juena logs
```

On the server, use `sudo ./juena prod logs`; Ctrl+C stops following logs.

The logs report each repository's revision as it is cloned or refreshed, then a
survey line naming any repository that is cloned but **not** in the store — that
is the one to take to juena-rag.

## Step 6: Ingest It Into juena-rag

Until this runs, the repository is readable file by file and invisible to
search. From this chatbot checkout (no sibling core/RAG checkout is needed):

```bash
./juena repos-ingest my-library --dry-run   # split the files and count only
./juena repos-ingest my-library             # embed and store
```

It reads the entry you added in Step 2; there is nothing to add in juena-rag.

For the server:

```bash
sudo ./juena prod repos-ingest my-library --dry-run
sudo ./juena prod repos-ingest my-library
sudo ./juena prod doctor
```

Ingestion uses Blablador for embeddings. Hugging Face supplies only the
tokenizer files used for local token counting; a successful anonymous download
does not need an HF token. Watch for the final result for each repository and
compare its nonzero chunk count in `rag.indexed`. `rag.cloned` means the working
tree exists. `needs_ingestion=false` means some indexed content exists, not that
all configured repositories are complete. A healthy RAG service can have an
empty or partial corpus.

Confirm it worked by asking JüNA which repositories it can search — that calls
`list_indexed_repositories`, which reports the store's own catalogue and chunk
counts rather than what happens to be cloned on disk.

## Choosing Good Settings

These apply to the entry in `config/repositories.yaml`, and take effect at its
next ingest:

- Small or medium Python repo: defaults are usually fine
- Monorepo: narrow the include globs to the relevant package or service
- Large generated trees: widen the exclude globs
- Docs-heavy repo: keep the documentation paths accurate so `search_docs_local`
  stays useful. Check they exist: jscatter's documentation is `README.rst` and
  `src/jscatter/doc/`, not `README.md` and `docs/`

## Adding Multiple Repositories

Just append more items under `repositories:`.

Example:

```yaml
repositories:
  - id: repo-a
    source:
      url: https://github.com/org/repo-a.git
      branch: main
    chunk_size: 3000
    chunk_overlap: 400

  - id: repo-b
    source:
      url: https://github.com/org/repo-b.git
      branch: master
    chunk_size: 3000
    chunk_overlap: 400
```

The agent will discover both through `list_repositories`.
Ingest both with `./juena repos-ingest repo-a repo-b`, or omit IDs to process
the entire registry sequentially. For example, if only jscatter has already
been ingested, add the seven remaining entries with:

```bash
sudo ./juena prod repos-ingest datreat mdanse sasview sasmodels sasdata silx vitess
sudo ./juena prod doctor
```

Do not run overlapping ingestion jobs. After a failure, retry the affected IDs;
unchanged successful entries are skipped unless `--force` is supplied.

## Updating an Existing Repo Later

There are two different cases:

1. You added a brand-new repo ID
2. You changed an already-indexed repo's branch, globs, or content and need a
   fresh semantic index

Case 1 is handled by the normal workflow above.

For case 2 there is no index on this side to delete and no cache to clear: run
`./juena repos-ingest <id>` (with `--force` if only settings changed). juena-rag
replaces the repository's chunks atomically, and search sees the new content at
once. The command then moves the *clone*, which is what `read_file` and `grep`
see, to the commit just ingested; if JüNA is not running, its next start does.

New commits upstream are not picked up automatically. Re-ingest every two weeks,
and at least once a month, or sooner after a release people will ask about:

```bash
sudo ./juena prod repos-ingest
sudo ./juena prod doctor
```

This processes the complete registry sequentially. Unchanged revisions are
skipped without embedding calls; changed repositories are re-embedded in full.
Search continues using each repository's old data until its replacement is
complete. The database volume retains the corpus across normal app rebuilds
and restarts. Locally, use `./juena repos-ingest` without `prod`.

## Useful Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `REPO_CONFIG_PATH` | `config/repositories.yaml` | Override the repo config file location |
| `JUENA_APP_ROOT` | Source checkout for native runs; `/app` in the image | Root for application files outside the installed Python package |
| `REPO_CACHE_DIR` | `data/repos` locally, `/data/repos` in Docker | Where repos are cloned |
| `JUENA_RAG_URL` | none — required | The juena-rag service that holds the vectors and answers every search |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| New repo does not appear in the registry | App did not restart with updated config | Rebuild with `./juena up`, or `sudo ./juena prod up` on the server |
| Registry warning names a path under `site-packages` | Installed package used the wrong application root | Deploy the path fix; the image uses `/app` and Compose sets `/app/config/repositories.yaml` |
| `git clone` fails | Bad URL or missing access | Check the URL and ensure the runtime has network/SSH credentials |
| Wrong branch error | `source.branch` is wrong | Check the repo's default branch and update the config |
| Service startup is slow | A large repository is cloning | Watch the bootstrap log and wait for the health check |
| Repo is listed but search finds nothing | It was never ingested into juena-rag | `./juena repos-ingest <id>`; `list_indexed_repositories` confirms |
| Search misses recent code | The repository has commits that were never ingested | `./juena repos-ingest`, or its server `prod` variant; restarting alone will not do it |
| Docs search misses files | `docs_paths` does not match the repo layout | Fix `docs_paths` in `config/repositories.yaml`, then `./juena repos-ingest <id> --force` |

## Short Version

If you just want the operator checklist:

1. Add a new item under `repositories:` in `config/repositories.yaml`
2. Set `id`, `source.url`, the correct `source.branch`, `chunk_size` and
   `chunk_overlap`
3. Fully restart the app
4. Confirm the repo shows up in `GET /repositories`
5. Run `./juena repos-ingest <id>` — **without this the repository can be read
   but not searched**
6. Ask JüNA which repositories it can search; the new one should be listed with
   a non-zero chunk count

For the server, use the `prod` command variants above. See
[deployment troubleshooting](deploy/TROUBLESHOOTING.md#rag-is-healthy-but-repositories-or-search-tools-are-missing)
if configured, cloned and indexed repositories disagree.
