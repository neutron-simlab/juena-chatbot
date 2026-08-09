"""Protocol constants shared by the sandbox broker and application client.

Deliberately dependency-free: ``broker_service`` runs as a standalone process on
the container host and must not import ``juena.core.config`` (or anything that
pulls in the application's ``.env`` loading) just to learn these limits.
"""

BROKER_AUTH_HEADER = "X-Juena-Sandbox-Token"

# Per-execution artifact budget. The broker enforces these while collecting from
# the workspace; the application enforces them again while registering, because
# the two run in different trust domains.
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

# What the broker picks up from a finished workspace: every downloadable type
# plus PNG, which is handled separately as an inline image rather than a
# download. Derived so that adding a format to DOWNLOAD_TYPES is a single edit.
COLLECTED_EXTENSIONS = frozenset(DOWNLOAD_TYPES) | {".png"}
