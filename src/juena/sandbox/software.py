"""Repository-backed software available in the fixed sandbox image."""

from __future__ import annotations

from importlib.resources import files


_REQUIREMENTS_RESOURCE = "repository-requirements.txt"


def repository_software_requirements() -> tuple[str, ...]:
    """Return the pinned, non-comment entries installed in the sandbox image."""

    content = files("juena.sandbox").joinpath(_REQUIREMENTS_RESOURCE).read_text(
        encoding="utf-8"
    )
    return tuple(
        line
        for raw_line in content.splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )


def repository_software_prompt() -> str:
    """Describe the repository software without implying runtime installation."""

    requirements = repository_software_requirements()
    if not requirements:
        return "No repository-backed software is preinstalled in the sandbox image."
    installed = ", ".join(f"`{requirement}`" for requirement in requirements)
    return (
        f"Repository-backed software preinstalled in the sandbox image: {installed}. "
        "Use these installed packages directly; runtime package installation remains forbidden."
    )
