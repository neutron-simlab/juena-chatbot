"""Small command policies shared by approval and execution middleware."""

from __future__ import annotations


def is_redundant_artifact_export(command: str) -> bool:
    """Return whether a command tries to print a stored output as base64."""
    normalized = " ".join(command.casefold().split())
    if "/workspace/outputs" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in ("base64", "b64encode", "data:image/")
    )


def requires_execution_approval(request: object) -> bool:
    """Skip approval only for exports the backend will refuse without running."""
    tool_call = getattr(request, "tool_call", {})
    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    command = args.get("command", "") if isinstance(args, dict) else ""
    return not is_redundant_artifact_export(str(command))
