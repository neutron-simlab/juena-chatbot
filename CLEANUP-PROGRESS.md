# Dead-code cleanup — progress and remaining work

Status as of 2026-08-09. Branch `update/backend-and-permission`.

**Test baseline:** started at 308 passed / 4 skipped. Currently **294 passed / 4 skipped, 0 failures**.
The drop is entirely removed tests for removed code (7 benchmark + 8 invoke-path + net −1 input-handler).

Resume by reading "Remaining work" below. Each step is independent; run
`.venv/bin/python -m pytest -q` after each.

---

## Decisions taken (do not re-litigate)

1. **Benchmarks** — delete entirely, including `evaluate` / `trectools` deps.
2. **Docs** — archive implemented planning docs into the gitignored `DOCS/`, not delete.
3. **Refactor depth** — delete + de-duplicate helpers + prune tests covering removed code.
4. **Non-streaming API** — remove the whole path (server endpoints *and* client methods).
5. **Compat shims** — drop agent aliases, deprecated `user_id`, and `LEGACY_SQLITE_ARCHIVE_DIR`.
6. **Search tools** — collapse `search_code_semantic` + `search_code_hybrid` into one tool.

**Out of scope:** `SANDBOX-CLEANUP.md` describes *future* work (podman-py migration, ID-based
approvals). Leave it untouched and unexecuted.

---

## DONE

### Step 0 — Track the sandbox work ✅
All previously-untracked work (`deploy/`, `src/juena/sandbox/`, `tests/test_sandbox_*.py`,
`SANDBOX-CLEANUP.md`, `SANDBOX-IMPLEMENTATION.md`) is committed as `439e351 update: first
implementation`. The cleanup diff now stands on its own.

### Step 1 — Deleted the `retrieval/` package ✅
- Deleted 4 pure re-export shims: `retrieval/{bootstrap,repo_config,repo_manager,vector_index}.py`.
- Deleted the dead benchmark cluster: `retrieval/benchmark_runner.py` (358 lines, zero refs),
  `retrieval/benchmark_dataset.py` (202 lines, test-only), `tests/test_benchmark_dataset.py`.
- Moved `retrieval/semantic_search.py` → **`src/juena/indexing/semantic_search.py`**
  and updated the one import in `src/juena/tools/repo_search.py`.
- `src/juena/retrieval/` no longer exists. `grep -rn "juena.retrieval"` → clean.

### Step 2 — Deleted the non-streaming API path ✅
**Server** (`src/juena/server/api/endpoints.py`, 748 → 528 lines): removed `invoke`,
`invoke_with_files`, `_invoke_with_attachments`, `get_agents`, `get_repositories`, plus the
now-dangling imports (`AIMessage`, `list_registered_agents`, `ChatMessage`, `UserInput`,
`langchain_to_chat_message`, `ChatbotServerError`).

**Client** (`src/juena/clients/client.py`): removed `invoke`, `ainvoke`, `astream`,
`list_agents`, `list_repositories`, `update_agent`, plus dangling `AsyncGenerator`/`UserInput`
imports.

**Tests:** removed 4 tests from `tests/test_agent_client.py` and 4 from
`tests/test_api_endpoints_files.py`. Rewrote the cross-user authorization assertion in
`tests/test_postgres_integration.py:203` from `POST /react_agent/invoke` onto `POST /stream`
(same `_authorize_thread` call, so coverage is preserved).

**Also:** deleted `AgentInputHandler.prepare_input` — a redundant fourth composition of
`build_run_context` + `build_input`, used only by tests. Rewrote
`tests/test_agent_input_handler.py` onto the two primitives production actually calls, so
coverage moved rather than disappeared (3 tests, passing).

---

## REMAINING WORK

### Step 3 — Retire expired compat shims (IN PROGRESS, not started)

| Target | Action |
|---|---|
| `src/juena/agents/juena_agent.py:165-166` | Delete `register_agent_alias("code_chat_agent", …)` and `("react_agent", …)`. Then delete the whole alias machinery in `src/juena/server/agent/registry.py`: `_agent_aliases` (line 33), `register_agent_alias` (67), `_resolve_agent_id` (82) and its call sites. Update the README note (~line 32). **Verified safe:** `agent_id` is *not* persisted in Postgres. |
| `src/juena/schema/server.py:27-34` | Delete `UserInput.user_id`. Also drop the "deprecated request user_id field is ignored" sentence from the `stream` docstring (`endpoints.py`, now ~line 319) and any twin in `resume`. |
| `LEGACY_SQLITE_ARCHIVE_DIR` | Delete from `src/juena/core/config.py:191`; `docker-compose.yml` env line ~122, the `:ro` mount ~53, and the `chatbot-db` volume ~179-181; and the assertions in `tests/test_config_paths.py` (lines ~28, ~53, ~91). Nothing reads it. |
| `src/juena/server/agent/runtime_model_middleware.py:47-69` | Delete `_provider_model_from_config`. Its only writer (`input_handler.py:58-63`) *always* also sets `RuntimeModelContext`, so the fallback is unreachable. |
| `src/juena/server/agent/runtime_model_middleware.py:85-88` | Delete the `state.get("llm_provider")` / `state.get("llm_model")` branch — nothing writes those state keys. |
| `src/juena/core/log.py:45` | Delete the unreachable "should never be reached" fallback. |
| `juena:183-185` | Delete `cmd_sandbox` — a vestigial one-line alias for `cmd_up --wait`. No doc references it. Then fix `env.example:52,62`, which describe it incorrectly (say `juena up`). |

### Step 4 — Delete speculative back-compat in `src/juena/sandbox/`

This package has never shipped, so these migration paths guard data that cannot exist.
Verified: `find data/artifacts -name '*.png'` → 0 results.

- `artifacts.py:222` — delete `ArtifactStore.register_png()` (self-described "compatibility
  wrapper", zero callers, superseded by `register_artifact`).
- `artifacts.py:160-163` — collapse `_content_path()` to
  `self._user_root(user_id) / f"{artifact_id}.bin"`, dropping the legacy `.png` fallback.
- `artifacts.py:276-277` — drop the `_load()` `kind` back-fill.
- `sandbox/__init__.py` — delete the `sandbox_enabled()` wrapper and `__all__`; every caller
  imports from `juena.sandbox.runtime`. **Keep the file itself** (docstring only) —
  `broker_service.py` runs standalone and must not import `juena.core.config`.

### Step 5 — Collapse the duplicated search tools

`src/juena/tools/repo_search.py` registers three tools (`build_code_chat_tools`, ~line 341),
but `_search_code_semantic` (131) and `_search_code_hybrid` (155) run the *same* dense vector
search — only the JSON serialization differs. `search_code_hybrid` is a misnomer from the
deleted sparse retriever; its docstring admits "it is NOT exact-string search".

Keep one semantic code-search tool plus `search_docs_local`. No `.md` prompt names these
tools, so blast radius is `repo_search.py`, `tests/test_repo_search_tools.py`, and the label
strings at `tests/test_chat_interface.py:297-303`.

### Step 6 — De-duplicate copy-pasted helpers

**`_find_workspace_root()` — 4 identical copies** in `indexing/vector_index.py:49`,
`bootstrap.py:21`, `repo_config.py:49`, `repo_manager.py:26`, plus a fifth `_find_repo_root()`
at `core/config.py:13`.

> **Constraint:** `core/config.py` calls `load_dotenv()` at import (line 33) — which is exactly
> why callers use a lazy `_get_config()` instead of importing `Config`. The shared helper must
> **not** live in `config.py`.

Add a side-effect-free `src/juena/core/paths.py` exposing `repo_root()`; point all five at it;
keep `_repo_data_dir()` in `config.py` delegating to it.

**`_get_config()` — 4 identical copies** in `tools/tavily.py:21`, `tools/context7.py:14`,
`core/llms_providers.py:26`, `indexing/vector_index.py:43`. Define `get_config()` once in
`core/paths.py`. It must stay lazy — that is its entire purpose.

> **Test impact (in scope):** 10 sites patch this per-module by name, e.g.
> `monkeypatch.setattr(tavily_module, "_get_config", lambda: StubConfig)` —
> `test_context7_tools.py` (3), `test_tavily_tools.py` (4), `test_vector_index.py` (3). Each
> module still binds the name locally so patching keeps working; only the attribute name
> changes `_get_config` → `get_config`.

**Sandbox duplicates:**
- `_context_value()` is byte-identical in `sandbox/runtime.py:23` and `sandbox/middleware.py:13`.
- Four artifact limits defined twice under different names: `artifacts.py:24-27`
  (`MAX_ARTIFACTS_PER_EXECUTION`, `MAX_ARTIFACT_BYTES`, `MAX_FILE_ARTIFACTS_PER_EXECUTION`,
  `MAX_FILE_ARTIFACT_BYTES`) vs `broker_service.py:37-40` (`MAX_ARTIFACTS`, `MAX_FILES`, same
  two byte limits). Move to `constants.py`, which both sides already import safely.
- Allowed extensions defined twice: `broker_service.py:41-57` `COLLECTED_EXTENSIONS` (13) ==
  `artifacts.py:33-46` `_DOWNLOAD_TYPES` keys (12) + `.png`. Single source in `constants.py`.
- Make `get_pending_interrupt` (`approvals.py:18`) and `SandboxIdentity` (`runtime.py:31`)
  private — neither has an external caller.

### Step 7 — Delete dead directories and dependencies

| Path | Why |
|---|---|
| `scripts/`, `examples/` | Contain **only** orphaned `__pycache__/*.pyc`; the `.py` sources are gone. Never tracked. |
| `skills/` | Empty. `tests/test_agent_resources.py:29` asserts the Dockerfile does *not* copy it. Keep the `.dockerignore` line `!src/**/skills/**/*.md` — that refers to per-specialist skills under `src/`. |
| `benchmarks/` | Gitignored, never tracked, **broken** (imports the deleted `juena.indexing.sparse_index`). |
| `.cursor/` | Empty, unreferenced. |
| `dist/`, all `__pycache__/`, `.pytest_cache/` | Generated. |

Extend the existing `cmd_clean` in `juena:317` rather than adding a new script.
Do **not** touch `data/`, `.sandbox/`, or `config/repositories.yaml`.

**Dependencies** in `pyproject.toml` — each verified with a repo-wide import grep:
drop `langchain-community` (line 33), `trectools` (line 44), `evaluate` (line 43, only used by
the already-deleted `benchmark_runner.py`). Then regenerate `uv.lock` with `uv sync`.

### Step 8 — Strip remaining unused symbols

- `server/streaming/handlers.py:14` — unused `AIMessage`.
- **Two stale "re-exported for callers" blocks** — both carry `# noqa: F401` and a comment
  claiming consumers that were verified not to exist:
  - `server/chat/input_constants.py:9-12` re-exports `ALLOWED_SUFFIXES`,
    `TEXT_READABLE_FILE_TYPES`. Its three importers (`server/utils.py`, `chat/input_utils.py`,
    `agent/input_handler.py`) take only `DISPLAY_*`, `ERROR_LINE_RE`, `FENCED_CODE_RE`,
    `LANGUAGE_SUFFIXES`, `UPLOADS_*`. **Delete the block.**
  - `server/chat/inputs.py:15-22` — drop `CHAT_INPUT_MAX_CHARS` and `TEXT_READABLE_FILE_TYPES`
    (zero internal uses, zero importers). Keep the other four, which are used in-file.

  `app/chat_interface.py` imports these from the canonical `schema/upload_limits.py`, so
  nothing regresses.
- `schema/llm_models.py` — `get_provider_for_model`, `is_valid_model_for_provider` (zero refs).
- `schema/server.py:118,128` — `ChatMessage.pretty_repr`, `pretty_print` (zero refs).
- `indexing/chunking.py:73-74` — `start_index`/`end_index`, self-described "leftovers from a
  sparse retriever that no longer exists". Confirm no index-schema bump is needed first.

### Step 9 — Archive implemented planning docs

Move into the gitignored `DOCS/`: `DEPLOYMENT-PLAN.md`, `PROMPT-AND-SKILLS-PLAN.md`,
`SANDBOX-PLAN.md`, `SANDBOX-IMPLEMENTATION.md`.

**Rescue these into tracked docs first — still operationally true:**
- `SANDBOX-IMPLEMENTATION.md:262-282` — the 9-item "Not yet executed" production gate →
  `deploy/SANDBOX.md`.
- `SANDBOX-PLAN.md:127-160` (threat model) and `:525-563` (acceptance criteria) →
  `deploy/SANDBOX.md`.
- `PROMPT-AND-SKILLS-PLAN.md:9-68` — already duplicated at `README.md:230-258`; verify, then discard.

`DEPLOYMENT-PLAN.md` needs no rescue — it contradicts what shipped (specifies 5 Compose
services incl. a Caddy gateway and an in-Compose `sandbox-broker`; the repo has 2 services and
a host-side broker over HTTP).

**Stays at root:** `README.md`, `SANDBOX-CLEANUP.md`, `SAML-IFFLOGIN.md`, `HOW-TO-ADD-REPO.md`.

**Fix while there:** the stale sentence at `SANDBOX-IMPLEMENTATION.md:106-107` claiming an
opt-in Docker Desktop engine (no such code path — `broker_service.py` only reads
`SANDBOX_PODMAN_BINARY`), and move the misplaced `SANDBOX_PODMAN_BINARY` comment out of
`env.example:65` into `deploy/sandbox-broker.env.example` where it already belongs.

---

## Final verification checklist

1. `.venv/bin/python -m pytest -q` — must stay green, no failures.
2. `.venv/bin/python -c "import juena.indexing.semantic_search, juena.sandbox.runtime, juena.server.service, juena.clients.client"`
3. These greps must all return nothing:
   `juena.retrieval`, `register_agent_alias`, `LEGACY_SQLITE_ARCHIVE_DIR`, `search_code_hybrid`.
4. `uv sync`, then re-run the suite (confirms nothing relied on the dropped deps transitively).
5. `./juena test`, then `./juena up` and `./juena health`.
6. **End-to-end smoke (tests cannot cover this):** with the stack up, confirm Streamlit on 9501
   and the API on 8080, then run one sandboxed code request that produces a plot — exercising
   approval → broker → artifact return after the `sandbox_enabled` / `_context_value` /
   artifact-constant consolidation.
7. `uv build && unzip -l dist/*.whl | grep repository-requirements` — `sandbox/software.py`
   loads that file via `importlib.resources`, and packaging relies on hatchling's default
   non-Python-file inclusion.
