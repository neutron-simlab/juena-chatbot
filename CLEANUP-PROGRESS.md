# Dead-code cleanup — completed

All nine steps are done. Branch `update/backend-and-permission`, eight commits
(`4c9b63e` .. `19da88a`) on top of the sandbox landing `439e351`.

**Net: 64 files changed, +468 / −2875 lines.**
**Tests: 291 passed, 4 skipped, 0 failures** (from a 308/4 baseline; the delta is
entirely tests for deleted code).

---

## What was removed

| Step | Outcome |
|---|---|
| 1 | `src/juena/retrieval/` deleted entirely — four re-export shims, plus a benchmark subsystem (560 lines) whose only entrypoint imported the long-deleted `sparse_index`. |
| 2 | The non-streaming API path: server `invoke`, `invoke_with_files`, `_invoke_with_attachments`, `GET /agents`, `GET /repositories`; client `invoke`, `ainvoke`, `astream`, `list_agents`, `list_repositories`, `update_agent`. `endpoints.py` went 748 → 528 lines. Also `AgentInputHandler.prepare_input`. |
| 3 | Expired compat shims: the `code_chat_agent`/`react_agent` aliases and the whole alias machinery; `UserInput.user_id`; the entire SQLite legacy archive (config var, compose env, `:ro` mount, `chatbot-db` volume); two unreachable paths in the runtime model middleware; the unreachable fallback in `core/log.py`; the vestigial `juena sandbox` subcommand. |
| 4 | Speculative back-compat in the never-shipped sandbox: `register_png`, the legacy `.png` content path, the `kind` back-fill, the `sandbox_enabled` re-export. Artifact limits and the collected-extension set consolidated into `constants.py`. |
| 5 | `search_code_hybrid` collapsed into `search_code_semantic` — they ran the same dense search. This orphaned `semantic_search()`, so `indexing/semantic_search.py` went too. |
| 6 | `_find_workspace_root()` (×5) and `_get_config()` (×4) consolidated into a new side-effect-free `juena/core/paths.py`. |
| 7 | Dropped `langchain-community`, `trectools`, `evaluate` and their transitive tail; deleted `scripts/`, `examples/`, `skills/`, `.cursor/`, `dist/`; extended `juena clean`. |
| 8 | Two stale "re-exported for callers" blocks whose callers did not exist, plus `AIMessage`, `is_valid_model_for_provider`, `get_provider_for_model`, `pretty_repr`, `pretty_print`. |
| 9 | Planning docs archived into gitignored `DOCS/`; the threat model and the nine-item production-enablement checklist rescued into the tracked `deploy/SANDBOX.md`. |

## Deliberately left alone

- **`SANDBOX-CLEANUP.md`** — proposes *future* work (podman-py migration, ID-based
  approval binding). Untouched and unexecuted, still at the repo root.
- **`benchmarks/`** — gitignored and its runner is broken, but it holds generated
  gold queries, `llm-generated-queries.yaml`, and a benchmark report that exist
  nowhere else. Deleting it is your call, not a mechanical cleanup. Nothing in
  `src/` references it any more.
- **`IndexChunk.start_index` / `end_index`** — the chunking docstring calls the
  *symbol-enrichment* fields sparse-retriever leftovers, not these. The character
  offsets are live: written into Chroma metadata, returned by `search`, and
  serialized into tool output. Removing them would force a full reindex for no gain.
- **`DOCS/SANDBOX-IMPLEMENTATION.md:106`** still claims an opt-in Docker Desktop
  engine that no code path implements. Left as-is — the file is now archived and
  explicitly unmaintained.

## Verification performed

- `./juena test` → 291 passed.
- Import smoke over `sandbox.runtime`, `sandbox.broker_service`, `server.service`,
  `clients.client`, `core.paths`, `indexing.vector_index`, `tools.repo_search`.
- Dead-reference greps all return zero: `juena.retrieval`, `register_agent_alias`,
  `LEGACY_SQLITE_ARCHIVE_DIR`, `search_code_hybrid`, `_find_workspace_root`,
  `register_png`.
- `uv sync` after the dependency removals, suite re-run green.
- `docker compose config` valid; `bash -n` clean on `juena` and `docker-entrypoint.sh`.
- `uv build` → wheel still ships `juena/sandbox/repository-requirements.txt`,
  which `software.py` loads via `importlib.resources`.

## Not verified — needs a running stack

The end-to-end smoke test was not run: bring the stack up (`juena up`), confirm
Streamlit on 9501 and the API on 8080, then run one sandboxed code request that
produces a plot. That exercises approval → broker → artifact return across the
`sandbox_enabled`, `_context_value`, and artifact-constant consolidation, which
unit tests do not cover.
