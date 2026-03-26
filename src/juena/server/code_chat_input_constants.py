"""Static constants and regexes for code-chat staged input handling."""

from __future__ import annotations

import re

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
INPUTS_PREFIX = "/inputs/"
UPLOADS_PREFIX = "/inputs/uploads/"
UPLOADS_MANIFEST_PATH = "/inputs/uploads_manifest.md"
FENCED_CODE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)\n(?P<body>.*?)```", re.DOTALL)
ERROR_LINE_RE = re.compile(
    r"(Traceback \(most recent call last\):|^\s*File \".*\", line \d+|^\s*at .+\(.+:\d+\)|"
    r"^\s*Caused by:|^\s*ERROR\b|\b[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b:|Error:)"
)
CODE_PATTERNS = (
    re.compile(r"^\s*(def|class|import|from|return|if|elif|else|for|while|try|except|with|async def)\b"),
    re.compile(r"^\s*(#include|public |private |protected |function\b|const\b|let\b|var\b|package\b|using\b|fn\b|impl\b)"),
    re.compile(r"^\s*[A-Za-z_][\w.]*\s*=\s*[^=]"),
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
