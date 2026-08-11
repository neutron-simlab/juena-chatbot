"""Sandbox API, persistence, and execution-evidence schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The two interrupt kinds that pause a turn. Defined here, next to the resume
# bodies that answer them, so the streaming layer can classify an interrupt
# without importing the agent package that raises it.
EXECUTE_APPROVAL_KIND = "execute_approval"
CLARIFICATION_KIND = "clarification"


class ArtifactRef(BaseModel):
    artifact_id: str
    filename: str
    mime_type: str
    kind: Literal["image", "file"]
    size: int
    width: int | None = None
    height: int | None = None
    caption: str
    created_at: datetime


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


class _ResumeBase(BaseModel):
    thread_id: str = Field(min_length=1, max_length=255)
    interrupt_id: str = Field(min_length=1, max_length=255)
    provider: str | None = None
    model: str | None = None


class ApprovalResumeInput(_ResumeBase):
    """Answer a sandbox `execute` approval.

    `kind` defaults to `execute_approval` so a client that predates the
    clarification interrupt keeps posting a valid body.
    """

    kind: Literal["execute_approval"] = "execute_approval"
    decision: Literal["approve", "edit", "reject"]
    edited_command: str | None = Field(default=None, max_length=100_000)


class ClarificationResumeInput(_ResumeBase):
    """Answer a question an agent raised with `ask_user`."""

    kind: Literal["clarification"]
    answer: str = Field(min_length=1, max_length=10_000)


ResumeInput = Annotated[
    ApprovalResumeInput | ClarificationResumeInput,
    Field(discriminator="kind"),
]
