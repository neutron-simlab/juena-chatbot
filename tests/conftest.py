"""Shared fixtures for code-chat-agent tests."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

# Unit tests must not inherit live integrations from the developer's .env.
# Individual sandbox tests enable the feature explicitly with local fakes.
os.environ["SANDBOX_ENABLED"] = "false"
os.environ["SANDBOX_STARTER_PREVIEW"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGFUSE_TRACING"] = "false"

# Configure juena_core once, for the whole suite, after the overrides above.
#
# Before the cutover this happened by accident: `juena.core.log` imported
# `Config` at module scope and `get_logger` was imported nearly everywhere, so
# importing almost any juena module read `.env`. Core deliberately does not do
# that -- `import juena_core` must work on a laptop with no environment at all
# -- so configuration is now an explicit act, performed by whoever starts a
# process. `main.py`, `juena.server.service` and `app/streamlit_app.py` each
# import this module; the test suite is the fourth entry point and says so here.
import juena.core.config  # noqa: E402,F401  calls juena_core.configure()

# After the configuration above, like every other juena import here.
from uuid import uuid4  # noqa: E402

from langfuse import Langfuse  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import juena_core.tracing as core_tracing  # noqa: E402

import juena.server.tracing as langfuse_tracing  # noqa: E402
from juena.core.config import Config  # noqa: E402


@pytest.fixture()
def tmp_repo(tmp_path: Path):
    """Create a tiny fake git repository and return its root."""
    repo = tmp_path / "origin"
    repo.mkdir()
    src = repo / "src"
    src.mkdir()
    (src / "main.py").write_text(textwrap.dedent("""\
        def greet(name: str) -> str:
            \"\"\"Return a greeting message.\"\"\"
            return f"Hello, {name}!"

        if __name__ == "__main__":
            print(greet("world"))
    """))
    (src / "utils.py").write_text(textwrap.dedent("""\
        import os

        def get_env(key: str, default: str = "") -> str:
            return os.getenv(key, default)
    """))
    (repo / "README.md").write_text("# Fake Repo\n\nA tiny test repository.\n")
    docs = repo / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n\nHow to use the greet function.\n")

    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True,
                   capture_output=True, env=env)
    return repo


@pytest.fixture()
def repo_config_path(tmp_path: Path, tmp_repo: Path):
    """Write a repositories.yaml pointing at *tmp_repo* via file:// URL."""
    cfg = {
        "repositories": [
            {
                "id": "fake-repo",
                "name": "Fake Repo",
                "description": "A tiny test repository",
                "source": {
                    "url": tmp_repo.as_uri(),
                    "branch": "main",
                },
                "include_globs": ["**/*.py", "**/*.md"],
                "exclude_globs": ["**/.git/**"],
                "max_file_bytes": 524288,
                "chunk_size": 200,
                "chunk_overlap": 20,
                "docs_paths": ["README.md", "docs/"],
            }
        ]
    }
    p = tmp_path / "repositories.yaml"
    p.write_text(yaml.dump(cfg))

    os.environ["REPO_CACHE_DIR"] = str(tmp_path / "clones")
    yield p
    os.environ.pop("REPO_CACHE_DIR", None)


class _LangfuseCapture:
    """The client `start_langfuse` was given, and the spans it exported."""

    def __init__(self, client: Langfuse, exporter: InMemorySpanExporter) -> None:
        self.client = client
        self.exporter = exporter

    def spans(self):
        self.client.flush()
        return self.exporter.get_finished_spans()


@pytest.fixture
def langfuse_capture(monkeypatch):
    """Start Langfuse the way the server does, exporting into memory.

    A fresh public key per test: the SDK keeps one set of resources per key
    for the life of the process, so a reused key would reuse an earlier
    test's exporter.
    """

    public_key = f"pk-lf-test-{uuid4().hex}"
    monkeypatch.setattr(Config, "LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setattr(core_tracing, "_hook", None)
    monkeypatch.setattr(langfuse_tracing, "_client", None)

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracing_enabled=True,
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    langfuse_tracing.start_langfuse(client)
    yield _LangfuseCapture(client, exporter)
    client.shutdown()
