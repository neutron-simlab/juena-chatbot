"""Dependency-free sandbox statuses and artifact limits."""

from typing import Literal

JobStatus = Literal["queued", "running", "completed", "failed", "timed_out"]

# A job occupies a slot while it is active. The partial unique indexes in the
# ORM derive their predicate from this tuple, so it is the single definition.
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "timed_out")

# Per-execution artifact budget. The worker selects filenames from its workspace;
# the application validates every file again before registering it.
MAX_ARTIFACTS_PER_EXECUTION = 4
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_FILE_ARTIFACTS_PER_EXECUTION = 8
MAX_FILE_ARTIFACT_BYTES = 10 * 1024 * 1024

# Downloadable (non-image) artifact types, mapped to the MIME type served back.
DOWNLOAD_TYPES = {
    ".csv": "text/csv",
    ".ipynb": "application/x-ipynb+json",
    ".json": "application/json",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".npy": "application/octet-stream",
    ".npz": "application/zip",
    ".pdf": "application/pdf",
    ".py": "text/x-python",
    ".sh": "text/x-shellscript",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
}

# What the worker picks up from a finished workspace: every downloadable type
# plus PNG, which is handled separately as an inline image rather than a
# download. Derived so that adding a format to DOWNLOAD_TYPES is a single edit.
COLLECTED_EXTENSIONS = frozenset(DOWNLOAD_TYPES) | {".png"}
