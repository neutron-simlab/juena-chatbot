"""Tool ownership for the neutron-research specialist."""

from __future__ import annotations

from typing import Any

FILESYSTEM_TOOL_DESCRIPTIONS = {
    "ls": (
        "List `/findings` to see what earlier specialists established, or `/inputs` "
        "for staged user files."
    ),
    "read_file": (
        "Read staged user inputs with line numbers. Use this for uploaded notes, "
        "tables, scripts, or experiment context supplied by the user."
    ),
    "grep": "Search staged user inputs for exact terms or values.",
    "glob": "Find staged files by a precise filename or extension pattern.",
    "write_file": (
        "Write short disposable scratch notes, or save what you established to "
        "`/findings/<topic>.md`, which the next specialist can read and nothing else "
        "carries. `/inputs` is read-only and this release has no execution workspace."
    ),
    "edit_file": (
        "Edit disposable scratch notes, or extend an existing `/findings/<topic>.md` "
        "rather than starting a second file on the same topic. `/inputs` is read-only."
    ),
}


def build_tools(tavily_tool: Any | None) -> list[Any]:
    """Expose Tavily only when configured."""

    return [tavily_tool] if tavily_tool is not None else []
