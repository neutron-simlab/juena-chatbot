"""Tool ownership for the software specialist."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from juena.tools.repo_search import build_code_chat_tools

FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List an absolute directory path. Use `/inputs` for staged user files. "
        "Use `/repos` only when indexed search cannot identify the repository or "
        "when the user asks which repositories exist."
    ),
    "read_file": (
        "Read an absolute file path with line numbers. Use it for staged inputs "
        "and exact evidence from the few best indexed repository search hits."
    ),
    "grep": (
        "Search for exact text. Use it only for symbols, imports, configuration "
        "keys, or call sites after indexed search identifies a likely area."
    ),
    "glob": (
        "Find precise filename, extension, test, documentation, or generated-file "
        "patterns. Do not use it for broad repository discovery."
    ),
    "write_file": (
        "Write only short disposable scratch notes. `/repos` and `/inputs` are "
        "read-only; return proposed code changes in the report."
    ),
    "edit_file": (
        "Edit only disposable scratch notes. `/repos` and `/inputs` are read-only."
    ),
}


def build_tools(
    repo_manager: Any,
    vector_index: Any,
    *,
    has_repo_research: bool,
    context7_tools: Sequence[Any],
) -> list[Any]:
    """Build only the software capabilities available in this deployment."""

    tools: list[Any] = []
    if has_repo_research:
        tools.extend(build_code_chat_tools(repo_manager, vector_index))
    tools.extend(context7_tools)
    return tools
