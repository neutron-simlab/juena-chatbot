"""Helpers for code-chat typed snippets and uploaded attachments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from deepagents.backends.utils import create_file_data, file_data_to_string
from fastapi import UploadFile
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from juena.core.log import get_logger

logger = get_logger(__name__)

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
_ALLOWED_SUFFIXES = {f".{suffix}" for suffix in TEXT_READABLE_FILE_TYPES}
_INPUTS_PREFIX = "/inputs/"
_UPLOADS_PREFIX = "/inputs/uploads/"
_UPLOADS_MANIFEST_PATH = "/inputs/uploads_manifest.md"
_FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)\n(?P<body>.*?)```", re.DOTALL)
_ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\):|^\s*File \".*\", line \d+|^\s*at .+\(.+:\d+\)|"
    r"^\s*Caused by:|^\s*ERROR\b|\b[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b:|Error:)"
)
_CODE_PATTERNS = (
    re.compile(r"^\s*(def|class|import|from|return|if|elif|else|for|while|try|except|with|async def)\b"),
    re.compile(r"^\s*(#include|public |private |protected |function\b|const\b|let\b|var\b|package\b|using\b|fn\b|impl\b)"),
    re.compile(r"^\s*[A-Za-z_][\w.]*\s*=\s*[^=]"),
)
_LANGUAGE_SUFFIXES = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "rust": ".rs",
    "rs": ".rs",
    "go": ".go",
    "shell": ".sh",
    "bash": ".sh",
    "sh": ".sh",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "toml": ".toml",
    "sql": ".sql",
    "html": ".html",
    "xml": ".xml",
    "css": ".css",
}


@dataclass(frozen=True)
class UploadedAttachment:
    """Validated attachment metadata staged in agent state."""

    original_filename: str
    staged_path: str
    char_count: int


@dataclass(frozen=True)
class PastedCodeContext:
    """Normalized code/error content extracted from the typed prompt."""

    user_goal: str
    code: str
    error_text: str
    language: str | None = None

    @property
    def contains_code(self) -> bool:
        return bool(self.code.strip())

    @property
    def contains_error(self) -> bool:
        return bool(self.error_text.strip())


@dataclass(frozen=True)
class PreparedCodeChatInputs:
    """Message override plus staged file updates for one code-chat turn."""

    message_override: str
    files_update: dict[str, Any | None]


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


def _line_is_error_like(line: str) -> bool:
    return bool(_ERROR_LINE_RE.search(line))


def _line_is_code_like(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return False
    if any(pattern.search(stripped) for pattern in _CODE_PATTERNS):
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
    fenced_blocks = list(_FENCED_CODE_RE.finditer(message))
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
        remaining = _FENCED_CODE_RE.sub("", remaining)

    error_text = ""
    if _ERROR_LINE_RE.search(remaining):
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
        suffix = _LANGUAGE_SUFFIXES.get(context.language.lower(), ".txt")
    return f"/inputs/current_code{suffix}"


async def list_existing_input_paths(
    agent: CompiledStateGraph,
    config: RunnableConfig,
) -> list[str]:
    """Return existing staged ``/inputs`` files for the current thread."""

    try:
        state: Any = await agent.aget_state(config=config)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to inspect existing staged inputs: %s", exc)
        return []

    values = getattr(state, "values", {}) or {}
    files = values.get("files", {}) or {}
    return sorted(
        path
        for path in files
        if isinstance(path, str) and path.startswith(_INPUTS_PREFIX)
    )


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


def _is_persistent_input_path(path: str) -> bool:
    return path.startswith(_UPLOADS_PREFIX) or path == _UPLOADS_MANIFEST_PATH


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

    existing_manifest = existing_files.get(_UPLOADS_MANIFEST_PATH)
    created_at = None
    if isinstance(existing_manifest, dict):
        created_at = existing_manifest.get("created_at")
        existing_manifest_text = file_data_to_string(existing_manifest).rstrip()
    else:
        existing_manifest_text = ""

    if existing_manifest_text and not new_uploads:
        return None

    if existing_manifest_text:
        start_index = len(existing_upload_paths) + 1
        appended_entries = [
            _format_upload_manifest_entry(
                start_index + index,
                staged_path=upload.staged_path,
                original_filename=upload.original_filename,
                char_count=upload.char_count,
            )
            for index, upload in enumerate(new_uploads)
        ]
        return "\n".join([existing_manifest_text, *appended_entries]), created_at

    entries: list[str] = []
    upload_by_path = {upload.staged_path: upload for upload in new_uploads}

    for index, path in enumerate(
        [*sorted(existing_upload_paths), *(upload.staged_path for upload in new_uploads if upload.staged_path not in existing_upload_paths)],
        start=1,
    ):
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

    if not entries:
        return None

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
                f"Inspect `{_UPLOADS_MANIFEST_PATH}` to see every upload in this thread.",
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
                f"- {_UPLOADS_MANIFEST_PATH}",
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


async def prepare_code_chat_turn_inputs(
    agent: CompiledStateGraph,
    config: RunnableConfig,
    message: str,
    attachments: Sequence[UploadFile] | None = None,
) -> PreparedCodeChatInputs | None:
    """Prepare staged ``/inputs`` files and a manifest for one code-chat turn."""

    existing_files = await get_existing_input_files(agent, config)
    existing_upload_paths = sorted(
        path for path in existing_files if path.startswith(_UPLOADS_PREFIX)
    )
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
    if uploaded or pasted_context is not None:
        files_update["/inputs/current_message.txt"] = create_file_data(message)
        if pasted_context and pasted_context.contains_code:
            files_update[staged_code_path(pasted_context)] = create_file_data(pasted_context.code)
        if pasted_context and pasted_context.contains_error:
            files_update["/inputs/current_error.txt"] = create_file_data(pasted_context.error_text)

    if uploaded or pasted_context is not None or has_thread_uploads:
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
