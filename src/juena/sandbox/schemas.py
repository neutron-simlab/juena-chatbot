"""Sandbox API and persistence schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


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
