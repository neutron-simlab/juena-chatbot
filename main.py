"""
Main entry point for the chatbot template server.

Before starting the server, import your agent registration to register
your agent factory with the template.
"""
import os
import uvicorn

# Import agent registration (registers react_agent as default)
import juena.agents.react_agent


def main():
    """Run the FastAPI server using uvicorn."""
    # Disable reload in production by default, enable via RELOAD env var
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "juena.server.service:app",  # Import string for reload support
        host="0.0.0.0",
        port=8000,
        reload=reload,  # Enable auto-reload only if RELOAD=true
        log_level="info"
    )


if __name__ == "__main__":
    main()
