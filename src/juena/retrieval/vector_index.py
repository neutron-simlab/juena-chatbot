"""Backward-compatible re-export – canonical location is juena.indexing.vector_index."""

from juena.indexing.vector_index import RepoVectorIndex  # noqa: F401

# Re-export private helpers used by tests.
from juena.indexing.vector_index import (  # noqa: F401
    _build_embedding_function,
    _get_config,
    _INDEX_META_SCHEMA_VERSION,
    _INDEX_SCHEMA_VERSION,
)

# Allow monkeypatching via this module's namespace.
import juena.indexing.vector_index as _canonical  # noqa: E402
logger = _canonical.logger
