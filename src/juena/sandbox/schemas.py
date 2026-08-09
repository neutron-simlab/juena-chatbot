"""Wire and persistence schemas shared by the API and sandbox broker."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class BrokerFile(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    content_base64: str = Field(max_length=140_000_000)


class StageInputsRequest(BaseModel):
    workspace_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    files: list[BrokerFile] = Field(max_length=64)


class UploadFilesRequest(BaseModel):
    workspace_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    files: list[BrokerFile] = Field(min_length=1, max_length=64)


class DownloadFilesRequest(BaseModel):
    workspace_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    paths: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(
        min_length=1,
        max_length=64,
    )


class BrokerFileResult(BaseModel):
    path: str
    content_base64: str | None = None
    error: str | None = None


class ExecuteRequest(BaseModel):
    workspace_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    command: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: int = Field(ge=1, le=3600)
    max_output_bytes: int = Field(ge=1, le=1_000_000)


class BrokerArtifact(BaseModel):
    filename: str
    content_base64: str


class ExecuteResult(BaseModel):
    job_id: str
    output: str
    exit_code: int | None = None
    truncated: bool = False
    status: Literal["completed", "timed_out", "failed"]
    artifacts: list[BrokerArtifact] = Field(default_factory=list)


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
