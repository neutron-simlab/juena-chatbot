# Record a build and deployment

Keep one private record per deployment under ignored `DOCS/` or in the approved
operations system. Do not put real hostnames, operator correspondence, installed
certificate fingerprints, backup locations or live diagnostic output into the
tracked runbooks. This guide describes the evidence to collect; it does not
claim that any particular server has passed acceptance.

## Record build provenance

Build from this chatbot checkout. `pyproject.toml` and `uv.lock` pin core/RAG Git
revisions; Docker supplies Node.js for core's packaged frontend. Sibling source
checkouts are unnecessary.

Before deployment, record the branch, commit, modified build inputs, dependency
pins and lockfile hash:

```bash
git status --short
git rev-parse HEAD
sha256sum uv.lock
sudo ./juena prod build
```

Record the built image ID, platform, size and resolved base-image digests. Use
the configured `JUENA_IMAGE` tag; the server template uses:

```bash
sudo docker image inspect juena-chatbot:dev-server --format '{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}'
```

An image ID is not a registry manifest digest. Preserve the reviewed image with
an immutable tag or archive it for recovery. A mutable image tag or a successful
build alone does not establish which image is running or prove SAML readiness.
After activation, record the running container's image ID using the same project,
Compose files and environment selection used for deployment.

## Record runtime acceptance

Run the appropriate checks after deployment:

```bash
sudo ./juena prod ps -a
sudo ./juena prod health
sudo ./juena prod doctor
sudo ./juena prod sandbox-test       # when execution is enabled
```

Record outcomes and evidence, not just that commands were invoked:

| Area | Evidence to retain privately |
| --- | --- |
| Configuration | Gateway mode, environment revision, installed paths and backup reference; no plaintext secrets in the report |
| Capacity | Host resources, service limits, measured workload peaks and any OOM/swap observations |
| Application | Running image ID, dependency pins, startup/schema checks, public API/UI health |
| SAML | Registration and trust confirmation plus the [browser checklist](SAML-IFFLOGIN.md#acceptance-checklist) |
| Retrieval | Each desired repository's indexed chunk count and revision; configured/cloned alone is insufficient |
| Sandbox | App flag, queue and worker connection, a passing queued PNG test, and image delivery in an authenticated chat |
| Shared gateway | Both applications remain reachable; record any routing change and its recovery result |
| Persistence | Database/workspace backups, restoration test and restart/reboot recovery |

Distinguish observations made directly from operator-supplied output. Include the
date, timezone and tested source/image. A partial ingestion run does not prove
all repositories are indexed; a worker lock does not prove a plot was delivered.

Use the [sandbox rollout checks](SANDBOX.md#broader-rollout-acceptance) for a wider
release. Tests run locally should record their environment and skipped checks;
they do not replace target-host or browser acceptance.

## Prepare rollback evidence

Preserve the previous source/image, compatible environment revision and installed
gateway configuration before changing them. Record database/schema compatibility
and backups separately: image rollback cannot undo a database migration.

`prod up` rebuilds from the chosen checkout; it is not an image-only rollback
command. Recover a reviewed source revision and compatible settings, or plan
an image-only recovery with the same project, Compose overlays and environment.
Keep the worker stopped until its dependencies match the recovered app.

Shared-gateway rollback restores routing only. Retain the existing application's
configuration and database recovery material independently. See
[SERVER.md](SERVER.md#capacity-persistence-and-rollback) and
[SHARED-PROXY.md](SHARED-PROXY.md#renewal-diagnostics-and-recovery).

## Suggested private record

- Deployment date/time and operator; host and public origin.
- Commit/branch, local modifications, lockfile hash and core/RAG pins.
- Built and running image IDs, base digests and protected image archive reference.
- Environment revision, gateway selection and protected backup references.
- Build/test results, including skipped or failed checks.
- API/UI, browser login/logout, repository and plotting acceptance results.
- Remaining work, recovery steps and restoration-test evidence.

Retain older records as historical evidence. Do not copy an earlier installation's
success status into a new one without repeating the relevant checks.
