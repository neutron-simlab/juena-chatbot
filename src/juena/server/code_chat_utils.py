"""Helper functions for code-chat staged input parsing and manifests."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Sequence

from deepagents.backends.utils import file_data_to_string

from juena.server.code_chat_input_constants import (
    CODE_PATTERNS,
    ERROR_LINE_RE,
    FENCED_CODE_RE,
    LANGUAGE_SUFFIXES,
    UPLOADS_MANIFEST_PATH,
    UPLOADS_PREFIX,
)
from juena.server.code_chat_input_types import PastedCodeContext, UploadedAttachment


def _line_is_error_like(line: str) -> bool:
    return bool(ERROR_LINE_RE.search(line))


def _line_is_code_like(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return False
    if any(pattern.search(stripped) for pattern in CODE_PATTERNS):
        return True
    if stripped.startswith((">>>", "...")):
        return True
    if stripped.startswith((" ", "\t")) and any(ch in stripped for ch in "=(){}[]:;"):
        return True
    if re.search(r"[{}();]", stripped):
        return True
    if re.match(r"^\s*</?[A-Za-z][^>]*>$", stripped):
        return True
    return False


def _extract_block(
    text: str,
    *,
    predicate,
    min_hits: int,
) -> str:
    lines = text.splitlines()
    best_content = ""
    current_lines: list[str] = []
    hits = 0

    for line in [*lines, "__END__"]:
        is_blank = line.strip() == ""
        if line != "__END__" and (predicate(line) or (current_lines and is_blank)):
            current_lines.append(line)
            if predicate(line):
                hits += 1
            continue

        if hits >= min_hits:
            candidate = "\n".join(current_lines).strip()
            if len(candidate) > len(best_content):
                best_content = candidate

        current_lines = []
        hits = 0

    return best_content


def _strip_first_occurrence(text: str, fragment: str) -> str:
    if not fragment:
        return text
    return text.replace(fragment, "", 1)


def _compact_summary(text: str, *, max_chars: int = 500) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(no typed prompt provided)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def extract_pasted_code_context(message: str) -> PastedCodeContext | None:
    """Best-effort extraction of inline code and errors from typed chat text."""

    if not message.strip():
        return None

    remaining = message
    fenced_blocks = list(FENCED_CODE_RE.finditer(message))
    code = "\n\n".join(
        block.group("body").strip("\n")
        for block in fenced_blocks
        if block.group("body").strip()
    ).strip()
    language = next(
        (block.group("lang").strip().lower() for block in fenced_blocks if block.group("lang").strip()),
        None,
    )
    if fenced_blocks:
        remaining = FENCED_CODE_RE.sub("", remaining)

    error_text = ""
    if ERROR_LINE_RE.search(remaining):
        error_text = _extract_block(remaining, predicate=_line_is_error_like, min_hits=1)
        if error_text:
            remaining = _strip_first_occurrence(remaining, error_text)

    if not code:
        code = _extract_block(remaining, predicate=_line_is_code_like, min_hits=3)
        if code:
            remaining = _strip_first_occurrence(remaining, code)

    if not code and not error_text:
        return None

    goal = "\n".join(line for line in remaining.splitlines() if line.strip()).strip()
    return PastedCodeContext(
        user_goal=goal,
        code=code,
        error_text=error_text,
        language=language,
    )


def staged_code_path(context: PastedCodeContext) -> str:
    """Return the canonical staged code path for extracted inline code."""

    suffix = ".txt"
    if context.language:
        suffix = LANGUAGE_SUFFIXES.get(context.language.lower(), ".txt")
    return f"/inputs/current_code{suffix}"


def is_persistent_input_path(path: str) -> bool:
    return path.startswith(UPLOADS_PREFIX) or path == UPLOADS_MANIFEST_PATH


def _format_upload_manifest_entry(
    index: int,
    *,
    staged_path: str,
    original_filename: str,
    char_count: int,
) -> str:
    return (
        f"{index}. `{staged_path}` "
        f"(from {original_filename}, {char_count} chars)"
    )


def build_uploads_manifest(
    existing_files: dict[str, Any],
    existing_upload_paths: Sequence[str],
    new_uploads: Sequence[UploadedAttachment],
) -> tuple[str, str | None] | None:
    """Create or update the persistent thread upload manifest."""

    existing_manifest = existing_files.get(UPLOADS_MANIFEST_PATH)
    created_at = None
    if isinstance(existing_manifest, dict):
        created_at = existing_manifest.get("created_at")
        existing_manifest_text = file_data_to_string(existing_manifest).rstrip()
    else:
        existing_manifest_text = ""

    if existing_manifest_text:
        if not new_uploads:
            return None
        start_index = len(existing_upload_paths) + 1
        new_entries = [
            _format_upload_manifest_entry(
                start_index + index,
                staged_path=upload.staged_path,
                original_filename=upload.original_filename,
                char_count=upload.char_count,
            )
            for index, upload in enumerate(new_uploads)
        ]
        return "\n".join([existing_manifest_text, *new_entries]), created_at

    upload_by_path = {upload.staged_path: upload for upload in new_uploads}
    manifest_paths = sorted(existing_upload_paths)
    manifest_paths.extend(
        upload.staged_path
        for upload in new_uploads
        if upload.staged_path not in existing_upload_paths
    )
    if not manifest_paths:
        return None

    entries: list[str] = []
    for index, path in enumerate(manifest_paths, start=1):
        upload = upload_by_path.get(path)
        if upload is not None:
            entries.append(
                _format_upload_manifest_entry(
                    index,
                    staged_path=upload.staged_path,
                    original_filename=upload.original_filename,
                    char_count=upload.char_count,
                )
            )
            continue

        file_data = existing_files.get(path)
        char_count = len(file_data_to_string(file_data)) if isinstance(file_data, dict) else 0
        entries.append(
            _format_upload_manifest_entry(
                index,
                staged_path=path,
                original_filename=Path(path).name,
                char_count=char_count,
            )
        )

    lines = [
        "# Persistent chat uploads",
        "",
        "Files uploaded in this chat remain available until the chat is deleted.",
        "",
        *entries,
    ]
    return "\n".join(lines), created_at


def build_inputs_manifest(
    raw_message: str,
    attachments: Sequence[UploadedAttachment],
    pasted_context: PastedCodeContext | None,
    *,
    has_thread_uploads: bool,
) -> str:
    """Build the short staged-input manifest injected into agent history."""

    summary = pasted_context.user_goal if pasted_context and pasted_context.user_goal else raw_message
    lines: list[str] = []

    if has_thread_uploads:
        lines.extend(
            [
                "Persistent uploaded files for this chat are available under `/inputs/uploads/`.",
                f"Inspect `{UPLOADS_MANIFEST_PATH}` to see every upload in this thread.",
            ]
        )

    if attachments or pasted_context is not None:
        lines.extend(
            [
                "This message also staged turn-scoped helper files under `/inputs`.",
                "Inspect `/inputs` before answering.",
            ]
        )
    elif has_thread_uploads:
        lines.append("Inspect `/inputs/uploads/` before answering if the user refers to uploaded material.")

    lines.extend(
        [
            "",
            f"User request: {_compact_summary(summary)}",
        ]
    )

    if has_thread_uploads:
        lines.extend(
            [
                "",
                "Persistent files:",
                f"- {UPLOADS_MANIFEST_PATH}",
            ]
        )

    if attachments or pasted_context is not None:
        lines.extend(
            [
                "",
                "Current turn files:",
                "- /inputs/current_message.txt",
            ]
        )

    if pasted_context and pasted_context.contains_code:
        lines.append(f"- {staged_code_path(pasted_context)}")
    if pasted_context and pasted_context.contains_error:
        lines.append("- /inputs/current_error.txt")

    for attachment in attachments:
        lines.append(
            f"- {attachment.staged_path} "
            f"(from {attachment.original_filename}, {attachment.char_count} chars)"
        )

    lines.extend(
        [
            "",
            "Use `/inputs` as primary evidence for the user's pasted or uploaded materials.",
            "Correlate them with `/repos` and Context7 only when helpful.",
        ]
    )
    return "\n".join(lines)
