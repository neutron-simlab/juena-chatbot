"""Request-scoped typed evidence for direct sandbox tool calls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxExecutionEvidence(BaseModel):
    """Facts captured before Deep Agents flattens an ExecuteResponse to text."""

    model_config = ConfigDict(extra="forbid")

    command: str
    status: Literal[
        "completed",
        "failed",
        "timed_out",
        "busy",
        "unavailable",
        "worker_error",
        "skipped",
        "tool_error",
    ]
    exit_code: int | None = None
    truncated: bool = False
    artifact_ids: list[str] = Field(default_factory=list)
    artifact_filenames: list[str] = Field(default_factory=list)
    dropped: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def attempted(self) -> bool:
        """Whether a command actually reached the worker.

        A `skipped` export was refused before running, so counting it as a
        failure would report a run as broken when nothing was even tried.
        """
        return self.status != "skipped"

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.exit_code == 0


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
