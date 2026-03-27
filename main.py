"""
Main entry point for the chatbot template server.

Before starting the server, import your agent registration to register
your agent factory with the template.
"""
import os
import uvicorn

from juena.core.config import global_config
from juena.indexing.bootstrap import BOOTSTRAP_DONE_ENV, bootstrap_repositories

# Import agent registrations – each module self-registers via register_agent_factory
import juena.agents.react_agent       # default agent
import juena.agents.code_chat_agent   # code-chat agent (hybrid search over repos)


def ensure_startup_bootstrap() -> None:
    """Run repository bootstrap once per process before starting the server."""
    if os.getenv(BOOTSTRAP_DONE_ENV) == "1":
        return

    bootstrap_repositories()
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
        log_level="info"
    )


if __name__ == "__main__":
    main()
