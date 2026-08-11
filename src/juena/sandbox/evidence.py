"""Request-scoped capture of typed evidence for direct sandbox tool calls.

The evidence itself is `juena.schema.sandbox.SandboxExecutionEvidence`; this
module owns only the context-local buffer that carries it from the backend,
where the exit code is still a number, up to the middleware that records it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from juena.schema.sandbox import SandboxExecutionEvidence

_capture: ContextVar[list[SandboxExecutionEvidence] | None] = ContextVar(
    "juena_sandbox_execution_capture",
    default=None,
)


@contextmanager
def capture_sandbox_execution() -> Iterator[list[SandboxExecutionEvidence]]:
    """Capture backend evidence produced by one direct execute tool call."""

    events: list[SandboxExecutionEvidence] = []
    token = _capture.set(events)
    try:
        yield events
    finally:
        _capture.reset(token)


def record_sandbox_execution(**values: Any) -> None:
    """Append evidence only when execution middleware opened a capture."""

    events = _capture.get()
    if events is not None:
        events.append(SandboxExecutionEvidence.model_validate(values))
