"""Sandbox API and persistence schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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


class ApprovalResumeInput(BaseModel):
    thread_id: str = Field(min_length=1, max_length=255)
    interrupt_id: str = Field(min_length=1, max_length=255)
    decision: Literal["approve", "edit", "reject"]
    edited_command: str | None = Field(default=None, max_length=100_000)
    provider: str | None = None
    model: str | None = None
