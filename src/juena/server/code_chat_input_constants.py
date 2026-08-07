"""Static constants and regexes for code-chat staged input handling."""

from __future__ import annotations

import re

# Canonical home is `juena.schema.upload_limits`, which the Streamlit UI also
# imports; re-exported here so server-side callers keep a single import site.
from juena.schema.upload_limits import (  # noqa: F401
    ALLOWED_SUFFIXES,
    TEXT_READABLE_FILE_TYPES,
)

INPUTS_PREFIX = "/inputs/"

# `additional_kwargs` keys on the staged human message. The manifest is what the
# model reads, but the checkpointer is the only store of message content, so the
# text the user actually typed has to ride along or a reloaded chat shows the
# manifest in the user's own bubble.
DISPLAY_TEXT_KEY = "juena_display_text"
DISPLAY_ATTACHMENTS_KEY = "juena_attachments"

UPLOADS_PREFIX = "/inputs/uploads/"
UPLOADS_MANIFEST_PATH = "/inputs/uploads_manifest.md"
FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)\n(?P<body>.*?)```", re.DOTALL)
# MULTILINE so the `^`-anchored alternatives match at the start of any line, not
# only at the start of the string -- without it, JS/Java stack traces went
# undetected. The trailing bare `Error:` alternative is deliberately absent: it
# matched ordinary prose such as "What does the Error: message mean?".
ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\):|^\s*File \".*\", line \d+|^\s*at .+\(.+:\d+\)|"
    r"^\s*Caused by:|^\s*ERROR\b|\b[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b:)",
    re.MULTILINE,
)
LANGUAGE_SUFFIXES = {
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
