"""Tool ownership for the software specialist."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from juena.tools.repo_search import build_code_chat_tools

FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List an absolute directory path. Use `/findings` to see what earlier "
        "specialists established, `/workspace` for shared files, and `/inputs` for pasted material. "
        "For repository content, use search_docs_local or search_code_semantic first. "
        "List a specific repository directory only to resolve a missing path after "
        "retrieval fails or is unavailable; do not walk repository trees. "
        "For which repositories exist, call `list_indexed_repositories` instead: "
        "`/repos` shows what is cloned on disk, not what is indexed."
    ),
    "read_file": (
        "Read an absolute file path with line numbers. Read `/skills` instructions and "
        "references directly. For repository questions, retrieve relevant paths with "
        "search_docs_local or search_code_semantic first, unless the user asked to "
        "inspect an exact file. Also reads shared workspace "
        "files, pasted inputs, and findings. Sufficient retrieved source excerpts need "
        "no additional file read. For missing context, read only the relevant "
        "range: offset=start_line-1 and limit=end_line-start_line+1 when supplied. "
        "Extend the range only if needed for context."
    ),
    "grep": (
        "Search for literal text, not regex, within a known file or directory. "
        "Use separate calls for different strings: `foo|bar` searches for those "
        "literal characters, not either name. Use for symbols, imports, configuration "
        "keys, or call sites in a relevant file identified by retrieval or an exact "
        "path supplied by the user or a skill. Do not use repository-wide grep as "
        "a substitute for indexed retrieval. Works on virtual `/repos` and `/skills` paths; "
        "shell grep cannot access those paths."
    ),
    "glob": (
        "Find a filename pattern within a specific directory, for shared files or "
        "a missing repository path after indexed retrieval fails or is unavailable. "
        "Use search_docs_local or search_code_semantic for repository discovery."
    ),
    "write_file": (
        "Create shared working files under `/workspace`; use `/workspace/.scratch` for disposable scripts. Write to "
        "`/findings/<topic>.md` only when the objective asks you to record findings; "
        "otherwise the report is the result. `/repos`, `/skills`, and `/inputs` "
        "are read-only."
    ),
    "edit_file": (
        "Edit the shared working file at its `/workspace` path, or extend an existing "
        "`/findings/<topic>.md` rather than starting a second file on the same topic. "
        "`/repos` and `/inputs` are read-only."
    ),
    "execute": (
        "Run scripts for a requested computation, simulation, or code test "
        "in the isolated sandbox; returns stdout/stderr and the exit code. "
        "Do not use it for file discovery, source inspection, text searches, "
        "line counts, or exploratory environment probes. Include necessary version "
        "checks in the requested script, not separate exploratory commands. "
        "The working directory is `/workspace/.scratch`; `/workspace` is writable "
        "and `/inputs` is mounted read-only. `/repos`, `/skills`, and `/findings` "
        "DO NOT exist in this shell. Use read_file, grep, glob, or ls for those paths, "
        "never shell commands, and do not retry a missing virtual path with another "
        "command. Installed packages can be imported normally. "
        "A documented how-to does not require execution. Do not install packages or "
        "access the network. Use absolute workspace paths and the optional timeout "
        "for commands that need it."
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
