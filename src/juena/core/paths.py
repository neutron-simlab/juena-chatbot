"""Repository-root resolution and lazy configuration access.

Deliberately free of import-time side effects. ``juena.core.config`` calls
``load_dotenv()`` when it is imported, so modules that only need a path -- or
that want the configuration read at call time rather than import time -- import
from here instead of reaching for ``Config`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Resolve the repository root by walking up until ``pyproject.toml``."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def get_config() -> Any:
    """Import ``Config`` lazily so importing a module does not load ``.env``.

    Also the seam tests patch to inject a stub configuration.
    """
    from juena.core.config import Config

    return Config
