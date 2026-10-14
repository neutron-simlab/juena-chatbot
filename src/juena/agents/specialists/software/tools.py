"""Tool ownership for the software specialist."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from juena.tools.repo_search import build_code_chat_tools

FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List an absolute directory path. Use `/findings` to see what earlier "
        "specialists established, `/workspace` for shared files, and `/inputs` for pasted material. "
        "Use `/repos` only when indexed search cannot identify the repository. "
        "For which repositories exist, call `list_indexed_repositories` instead: "
        "`/repos` shows what is cloned on disk, not what is indexed."
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
        "Create shared working files under `/workspace`; use `/workspace/.scratch` for disposable scripts. Write to "
        "`/findings/<topic>.md` only when the objective asks you to record findings; "
        "otherwise the report is the result. `/repos` and `/inputs` are read-only tool "
        "routes, not execution paths."
    ),
    "edit_file": (
        "Edit the shared working file at its `/workspace` path, or extend an existing "
        "`/findings/<topic>.md` rather than starting a second file on the same topic. "
        "`/repos` and `/inputs` are read-only."
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
