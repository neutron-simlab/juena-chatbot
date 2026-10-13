"""Send every agent run to the self-hosted Langfuse (deploy/langfuse/).

Switched on by LANGFUSE_TRACING. Core attaches whatever `_run_tracing` returns
to each run it starts (`juena_core.tracing`); the research runner asks for the
same thing for its background jobs.

What a trace is: one graph invocation. A new message, the answer to an
`ask_user` question or a sandbox approval, and each research job are separate
traces, grouped in Langfuse by session = the chat's thread id and by user.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from juena_core.tracing import RunTracing, configure_tracing

from juena.core.config import Config

__all__ = ["langfuse_lifespan", "start_langfuse"]

_client: Langfuse | None = None


def start_langfuse(client: Langfuse | None = None) -> None:
    """Create the Langfuse client and hand core the per-run hook.

    Call before the app is built, so the very first run carries the handler.
    The client is built from `Config` explicitly rather than letting the SDK
    read its own environment variables: one place decides where traces go.
    Tests pass a client wired to an in-memory exporter instead.
    """

    global _client
    _client = client or Langfuse(
        public_key=Config.LANGFUSE_PUBLIC_KEY,
        secret_key=Config.LANGFUSE_SECRET_KEY,
        base_url=Config.LANGFUSE_BASE_URL,
        tracing_enabled=True,
    )
    configure_tracing(_run_tracing)


def _run_tracing(*, thread_id: str, user_id: str) -> RunTracing:
    """Give one invocation its own handler and its Langfuse session and user.

    A new handler per invocation, so no handler state is shared between two
    users' requests, and an answer to an interrupt reliably starts a new trace
    in the same session rather than sometimes continuing an old one.

    `run_inline = True` is what puts the session and user on *every* span, not
    only the root. Without it, during async runs LangChain calls this
    (synchronous) handler on a worker thread, each call inside a throwaway copy
    of the context. The session and user the root's callback sets on the
    OpenTelemetry context vanish with that copy, so child chains and model
    generations arrive unattributed and Langfuse's per-user and per-session
    totals undercount. tests/test_langfuse_tracing.py guards this.
    """

    handler = CallbackHandler(public_key=Config.LANGFUSE_PUBLIC_KEY)
    handler.run_inline = True
    return RunTracing(
        callbacks=(handler,),
        metadata={"langfuse_session_id": thread_id, "langfuse_user_id": user_id},
    )


@asynccontextmanager
async def langfuse_lifespan() -> AsyncGenerator[None, None]:
    """Flush buffered spans when the server stops.

    uvicorn runs this only after the open requests have ended, and a chat
    request ends only once its graph has finished cancelling its nodes (core's
    `AgentStreamResponse`). Entered first among the extra lifespans, it also
    unwinds last: by then research has been cancelled and the sandbox stopped,
    and their final spans are in the buffer too. Spans still buffered when the
    process is killed rather than stopped are lost.
    """

    try:
        yield
    finally:
        if _client is not None:
            # A blocking HTTP flush; keep it off the event loop.
            await asyncio.to_thread(_client.shutdown)
