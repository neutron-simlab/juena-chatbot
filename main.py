"""
Main entry point for the chatbot template server.

Agent registration lives in `juena.server.service`, so it happens for every
process that serves the app -- including `uvicorn juena.server.service:app`,
which never imports this file.
"""
import os
import uvicorn

from juena.core.config import global_config
from juena.indexing.bootstrap import BOOTSTRAP_DONE_ENV, clone_repositories


def ensure_startup_bootstrap() -> None:
    """Clone the repositories once per process before starting the server.

    No index is built here: juena-rag holds the vectors. The clone is still
    needed because the model opens the real file to quote real line numbers.
    """
    if os.getenv(BOOTSTRAP_DONE_ENV) == "1":
        return

    clone_repositories()
    os.environ[BOOTSTRAP_DONE_ENV] = "1"


def main():
    """Run the FastAPI server using uvicorn."""
    # Disable reload in production by default, enable via RELOAD env var
    reload = os.getenv("RELOAD", "false").lower() == "true"

    ensure_startup_bootstrap()

    uvicorn.run(
        "juena.server.service:app",  # Import string for reload support
        host=global_config.SERVER_HOST,
        port=global_config.SERVER_PORT,
        reload=reload,  # Enable auto-reload only if RELOAD=true
        log_level="info",
        # On SIGTERM uvicorn waits for in-flight requests before it runs the
        # shutdown lifespans (research cancellation, Langfuse flush). Chat
        # streams are not what it waits for -- sse-starlette ends every event
        # stream as soon as shutdown begins -- but any other slow request
        # could use up Docker's whole grace period, and the container would be
        # killed before anything was flushed. 10 s caps that wait;
        # docker-compose.yml's stop_grace_period budgets for it.
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
