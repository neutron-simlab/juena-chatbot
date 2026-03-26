"""Helpers for code-chat typed snippets and uploaded attachments."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Sequence

from deepagents.backends.utils import create_file_data
from fastapi import UploadFile
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger
from juena.server.code_chat_input_constants import (
    ALLOWED_SUFFIXES as _ALLOWED_SUFFIXES,
    INPUTS_PREFIX as _INPUTS_PREFIX,
    TEXT_READABLE_FILE_TYPES,
    UPLOADS_MANIFEST_PATH as _UPLOADS_MANIFEST_PATH,
    UPLOADS_PREFIX as _UPLOADS_PREFIX,
)
from juena.server.code_chat_input_types import (
    PreparedCodeChatInputs,
    UploadedAttachment,
)
from juena.server.code_chat_utils import (
    build_inputs_manifest,
    build_uploads_manifest,
    extract_pasted_code_context,
    is_persistent_input_path as _is_persistent_input_path,
    staged_code_path,
)

logger = get_logger(__name__)

CHAT_INPUT_MAX_CHARS = 4_000
CHAT_INPUT_MAX_UPLOAD_MB = 1
MAX_ATTACHMENTS_PER_MESSAGE = 5
MAX_ATTACHMENT_BYTES = CHAT_INPUT_MAX_UPLOAD_MB * 1024 * 1024


def sanitize_uploaded_filename(filename: str | None) -> str:
    """Return a safe leaf filename for staging under ``/inputs/uploads``."""

    original = Path(filename or "upload.txt").name
    if original in {"", ".", ".."}:
        original = "upload.txt"

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", original).lstrip(".")
    if not sanitized:
        sanitized = "upload.txt"
    return sanitized


def is_text_readable_filename(filename: str | None) -> bool:
    """Return whether *filename* uses an allowed text-readable extension."""

    return Path(filename or "").suffix.lower() in _ALLOWED_SUFFIXES


def _dedupe_staged_name(filename: str, taken_names: set[str]) -> str:
    stem = Path(filename).stem or "upload"
    suffix = Path(filename).suffix
    candidate = filename
    index = 2

    while candidate in taken_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1

    taken_names.add(candidate)
    return candidate


def _decode_upload_text(raw: bytes, filename: str) -> str:
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"File '{filename}' exceeds the {CHAT_INPUT_MAX_UPLOAD_MB} MB per-file limit."
        )

    decoded = raw.decode("utf-8-sig", errors="replace")
    if "\x00" in decoded or "\ufffd" in decoded:
        raise ValueError(
            f"File '{filename}' must be a valid UTF-8 text file."
        )
    return decoded


async def normalize_uploaded_attachments(
    attachments: Sequence[UploadFile] | None,
    *,
    existing_upload_paths: Sequence[str] | None = None,
) -> tuple[list[UploadedAttachment], dict[str, Any]]:
    """Validate uploads and stage them under ``/inputs/uploads``."""

    uploads = list(attachments or [])
    if len(uploads) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise ValueError(
            f"At most {MAX_ATTACHMENTS_PER_MESSAGE} files may be uploaded per message."
        )

    files_update: dict[str, Any] = {}
    normalized: list[UploadedAttachment] = []
    taken_names = {
        Path(path).name
        for path in (existing_upload_paths or [])
        if isinstance(path, str) and path.startswith(_UPLOADS_PREFIX)
    }

    for index, upload in enumerate(uploads, start=1):
        original_name = upload.filename or f"upload_{index}.txt"
        sanitized_name = sanitize_uploaded_filename(original_name)

        if not is_text_readable_filename(sanitized_name):
            raise ValueError(
                f"File '{original_name}' has an unsupported extension. "
                "Only text-readable code, config, docs, notebook, and log files are allowed."
            )

        staged_name = _dedupe_staged_name(sanitized_name, taken_names)
        raw = await upload.read()
        text = _decode_upload_text(raw, original_name)
        staged_path = f"{_UPLOADS_PREFIX}{staged_name}"
        files_update[staged_path] = create_file_data(text)
        normalized.append(
            UploadedAttachment(
                original_filename=original_name,
                staged_path=staged_path,
                char_count=len(text),
            )
        )
        await upload.close()

    return normalized, files_update


async def get_existing_input_files(
    agent: CompiledStateGraph,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Return the current staged ``/inputs`` files for the active thread."""

    try:
        state: Any = await agent.aget_state(config=config)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to inspect existing staged inputs: %s", exc)
        return {}

    values = getattr(state, "values", {}) or {}
    files = values.get("files", {}) or {}
    return {
        path: file_data
        for path, file_data in files.items()
        if isinstance(path, str) and path.startswith(_INPUTS_PREFIX)
    }


async def prepare_code_chat_turn_inputs(
    agent: CompiledStateGraph,
    config: RunnableConfig,
    message: str,
    attachments: Sequence[UploadFile] | None = None,
) -> PreparedCodeChatInputs | None:
    """Prepare staged ``/inputs`` files and a manifest for one code-chat turn."""

    existing_files = await get_existing_input_files(agent, config)
    existing_upload_paths = sorted(path for path in existing_files if path.startswith(_UPLOADS_PREFIX))
    files_update: dict[str, Any | None] = {
        path: None
        for path in existing_files
        if not _is_persistent_input_path(path)
    }

    uploaded, uploaded_updates = await normalize_uploaded_attachments(
        attachments,
        existing_upload_paths=existing_upload_paths,
    )
    files_update.update(uploaded_updates)
    has_thread_uploads = bool(existing_upload_paths or uploaded)

    uploads_manifest = build_uploads_manifest(existing_files, existing_upload_paths, uploaded)
    if uploads_manifest is not None:
        manifest_content, manifest_created_at = uploads_manifest
        files_update[_UPLOADS_MANIFEST_PATH] = create_file_data(
            manifest_content,
            created_at=manifest_created_at,
        )

    pasted_context = extract_pasted_code_context(message)
    has_turn_inputs = bool(uploaded or pasted_context is not None)
    if has_turn_inputs:
        files_update["/inputs/current_message.txt"] = create_file_data(message)
        if pasted_context and pasted_context.contains_code:
            files_update[staged_code_path(pasted_context)] = create_file_data(pasted_context.code)
        if pasted_context and pasted_context.contains_error:
            files_update["/inputs/current_error.txt"] = create_file_data(pasted_context.error_text)

    if has_turn_inputs or has_thread_uploads:
        return PreparedCodeChatInputs(
            message_override=build_inputs_manifest(
                message,
                uploaded,
                pasted_context,
                has_thread_uploads=has_thread_uploads,
            ),
            files_update=files_update,
        )

    if files_update:
        return PreparedCodeChatInputs(
            message_override=message,
            files_update=files_update,
        )

    return None
