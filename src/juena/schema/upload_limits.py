"""Attachment limits shared by the Streamlit UI and the FastAPI server.

Deliberately dependency-free -- no FastAPI, deepagents or langgraph imports.
The UI needs these to refuse a bad upload *before* the message enters the
transcript, and importing them from ``juena.server`` used to drag the whole
server stack into the Streamlit process.

The server re-exports everything here and remains the enforcing side: UI checks
are a courtesy to the user, not a security boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

CHAT_INPUT_MAX_CHARS = 4_000
CHAT_INPUT_MAX_UPLOAD_MB = 1
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_ATTACHMENT_BYTES = CHAT_INPUT_MAX_UPLOAD_MB * 1024 * 1024

TEXT_READABLE_FILE_TYPES = [
    "py",
    "js",
    "ts",
    "java",
    "c",
    "cpp",
    "rs",
    "go",
    "sh",
    "md",
    "txt",
    "log",
    "json",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "csv",
    "sql",
    "html",
    "css",
    "xml",
    "ipynb",
]

ALLOWED_SUFFIXES = {f".{suffix}" for suffix in TEXT_READABLE_FILE_TYPES}


def is_text_readable_filename(filename: str | None) -> bool:
    """Return whether *filename* uses an allowed text-readable extension."""

    return Path(filename or "").suffix.lower() in ALLOWED_SUFFIXES


def validate_attachments(
    attachments: Sequence[tuple[str, int | None]],
) -> list[str]:
    """Return human-readable reasons *attachments* would be rejected.

    Takes ``(filename, size_in_bytes)`` pairs; an unknown size is skipped rather
    than assumed valid, since the server checks the real bytes either way. An
    empty list means the upload passes every check the UI can make locally.
    """

    errors: list[str] = []

    if len(attachments) > MAX_ATTACHMENTS_PER_MESSAGE:
        errors.append(
            f"At most {MAX_ATTACHMENTS_PER_MESSAGE} files may be uploaded per message "
            f"({len(attachments)} selected)."
        )

    for filename, size in attachments:
        if not is_text_readable_filename(filename):
            errors.append(
                f"'{filename}' has an unsupported extension. Only text-readable code, "
                "config, docs, notebook, and log files are allowed."
            )
        if size is not None and size > MAX_ATTACHMENT_BYTES:
            errors.append(
                f"'{filename}' is {size / 1024 / 1024:.1f} MB, over the "
                f"{CHAT_INPUT_MAX_UPLOAD_MB} MB per-file limit."
            )

    return errors
