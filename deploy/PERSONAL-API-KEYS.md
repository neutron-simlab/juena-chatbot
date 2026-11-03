# Personal API keys

IFFLogin supplies the user's identity, display name and email. Each user opens
**My profile** to manage their own keys. Blablador is required before a message or
approval resume can run. Tavily and Context7 are optional; their tools are absent
for users without a readable key. There is no shared chat or tool key fallback.

The initial screen and profile link to the
[Blablador API access guide](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/).
Users create a Helmholtz Codebase personal access token with `read_user` scope,
then select **Check and save**. The guide states a maximum token lifetime of one
year. A token is separate from the IFFLogin session. Removing it here deletes
JüNA's copy; revocation happens at the provider.

## Server configuration

Generate `JUENA_SECRET_KEY` once, store it in the protected server environment,
and back it up separately from PostgreSQL:

```bash
python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'
```

The command prints a new secret: copy it directly into your protected environment,
not into Git or a deployment log. Keep the key stable across restarts and replicas.
Changing it without re-encrypting existing credentials makes them unreadable; users
must then save replacements. Encryption protects a database copy without this key,
not an operator with access to both.

Set `RAG_BLABLADOR_API_KEY` to the dedicated service credential for repository
indexing and query embeddings. Compose maps it to RAG's existing
`BLABLADOR_API_KEY` setting. Production Compose does not forward this credential
to the chatbot. Standalone
juena-rag installations keep their existing `BLABLADOR_API_KEY` configuration.

Model IDs, `BLABLADOR_BASE_URL`, `CONTEXT7_MCP_URL`, timeouts, database credentials,
SAML keys, tracing and sandbox settings remain operator configuration. Service
URLs must use HTTPS. Personal keys are stored in PostgreSQL, never in `.env`.

## Upgrade an existing installation

1. Build with the updated, pinned juena-core revision. Publish that core commit
   before building from a remote checkout.
2. Back up the existing database. Add the persistent `JUENA_SECRET_KEY` and rename
   the server's embedding credential setting to `RAG_BLABLADOR_API_KEY`. Remove
   obsolete shared chat/tool key settings from the chatbot environment.
3. Deploy the updated application normally. Startup creates the additive
   `user_api_credentials` table. No database reset or existing-table alteration is
   needed; users, sessions, chats and workspace identities are preserved.
4. Existing users log in and add their personal Blablador key. Existing chats and
   files remain accessible before key setup. Personal keys are never imported
   automatically from the old shared environment.
5. Verify two IFFLogin users independently: save, replace and remove keys; check
   their display name/email; send a chat; exercise optional tools; sign out and
   sign in again. Check RAG retrieval separately using its server credential.

A key change applies to the next outbound model/tool request, including requests
inside background research. Requests already sent can finish. Logout keeps the
existing background-job lifecycle; removing Blablador's key causes subsequent
model requests to fail with a profile setup message. Provider rate limits and
outages do not delete saved keys. Changing a configured service endpoint requires
users to validate their keys again before sending them to that endpoint.

Rolling back the application leaves the additive credential table intact. Keep
the encryption key in the operator backup even if an older version ignores it.
An older release uses its old shared-key policy and needs its matching environment;
rollback does not silently restore old credentials.

## API and runtime boundary

`GET /auth/me` adds `chat_ready` and a `credentials` map. Each service reports
`saved`, `available`, `endpoint`, and `validated_at`; availability means the saved
key is readable and bound to the current endpoint, not a continuous provider
health check. Provider checks happen only on explicit saves and actual requests.

`PUT /auth/me/credentials/{service}` takes `{ "key": "…", "endpoint": "…" }`.
`DELETE` on the same path takes the displayed `endpoint` as a query parameter.
Supported services are `blablador`, `tavily`, and `context7`. Both require a live
session and the configured UI/API Origin. No route accepts another user's ID or
returns a saved key. The Streamlit client supplies its configured Origin.

Blablador validation uses `/models`; Tavily uses `/usage`; Context7 makes one
bounded public-library lookup (which may count toward its request quota). These
checks send no conversation content. Invalid replacements leave the old key intact.

Core accepts an async Blablador key supplier; the app resolves the authenticated
runtime user before every request, including fallbacks and summarizers. Cached
graphs store the supplier, not a user's key. Tool wrappers similarly create
request-owned authenticated clients. Core's optional invocation guard and error
mapper keep existing consumers compatible. Neither credentials nor encryption
keys are stored in prompts, graph state, checkpoint metadata or background job
inputs.
