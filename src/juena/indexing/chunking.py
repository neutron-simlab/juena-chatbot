"""
Chunking abstraction with a Chonkie adapter.

Provides a normalized ``IndexChunk`` model that the rest of the codebase
depends on. Chonkie is the parser-backed backend for Python and Markdown;
everything else uses ``RecursiveCharacterTextSplitter``, with language-aware
separators wherever langchain ships them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from juena.core.log import get_logger

logger = get_logger(__name__)

# Increment whenever chunk boundaries or offset semantics change. The vector
# index fingerprints this value so an existing collection is rebuilt once.
CHUNKING_VERSION = 2

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cl": "c",
    ".f": "fortran",
    ".F": "fortran",
    ".for": "fortran",
    ".FOR": "fortran",
    ".f90": "fortran",
    ".F90": "fortran",
    ".f95": "fortran",
    ".F95": "fortran",
    ".f03": "fortran",
    ".F03": "fortran",
    ".inc": "fortran",
    ".INC": "fortran",
}

_CHONKIE_LANGUAGES = {"python", "markdown"}

# Languages for which langchain ships syntax-aware separators. Anything absent
# here (notably Fortran, which has no Language member) uses the plain
# recursive splitter.
_LANGCHAIN_LANGUAGE_BY_NAME: dict[str, Language] = {
    "python": Language.PYTHON,
    "markdown": Language.MARKDOWN,
    "rst": Language.RST,
    "c": Language.C,
    "cpp": Language.CPP,
}


@dataclass
class IndexChunk:
    """Normalized chunk emitted by the chunking layer.

    Fields for symbol/path-token enrichment were removed because they were
    leftovers from a sparse retriever that no longer exists. Character offsets
    remain part of the persisted retrieval contract.
    """

    text: str
    file_path: str
    chunk_index: int
    content_hash: str
    language: str
    is_doc: bool
    start_index: int = 0
    end_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _detect_language(rel_path: str) -> str:
    ext = Path(rel_path).suffix
    return _LANGUAGE_BY_EXT.get(ext, "other")


# ------------------------------------------------------------------
# Chonkie adapter
# ------------------------------------------------------------------

def _chonkie_chunk(
    source: str,
    language: str,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Use Chonkie's CodeChunker for parser-backed splitting."""
    try:
        from chonkie import CodeChunker
    except ImportError:
        logger.warning("chonkie not installed – falling back to generic splitter")
        return []

    chonkie_lang = language if language != "markdown" else "markdown"
    try:
        chunker = CodeChunker(
            language=chonkie_lang,
            tokenizer="character",
            chunk_size=chunk_size,
        )
        raw_chunks = chunker.chunk(source)
    except Exception:
        logger.debug("Chonkie chunking failed for language %s – falling back", language)
        return []

    return [
        {
            "text": c.text,
            "start_index": c.start_index,
            "end_index": c.end_index,
            "token_count": c.token_count,
        }
        for c in raw_chunks
    ]


# ------------------------------------------------------------------
# Generic fallback
# ------------------------------------------------------------------

def _make_splitter(
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Build a splitter, syntax-aware when langchain knows the language.

    ``add_start_index`` makes the splitter report each chunk's exact offset in
    the source, which is the only reliable way to get it: searching for the
    chunk text afterwards fails whenever the splitter strips whitespace.
    """
    kwargs: dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "length_function": len,
        "add_start_index": True,
    }

    lc_language = _LANGCHAIN_LANGUAGE_BY_NAME.get(language)
    if lc_language is not None:
        return RecursiveCharacterTextSplitter.from_language(lc_language, **kwargs)
    return RecursiveCharacterTextSplitter(**kwargs)


def _generic_chunk(
    source: str,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    splitter = _make_splitter(language, chunk_size, chunk_overlap)
    return [
        {
            "text": doc.page_content,
            "start_index": doc.metadata.get("start_index", 0),
            "end_index": doc.metadata.get("start_index", 0) + len(doc.page_content),
            "token_count": len(doc.page_content),
        }
        for doc in splitter.create_documents([source])
    ]


def _split_oversized_chunks(
    chunks: list[dict[str, Any]],
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Apply a hard size cap after parser-backed chunking."""
    capped: list[dict[str, Any]] = []
    safe_overlap = min(chunk_overlap, max(chunk_size - 1, 0))

    for chunk in chunks:
        text = chunk["text"]
        token_count = chunk.get("token_count", len(text))
        if len(text) <= chunk_size and token_count <= chunk_size:
            capped.append(chunk)
            continue

        parent_start = chunk.get("start_index", 0)
        for child in _generic_chunk(text, language, chunk_size, safe_overlap):
            capped.append({
                **child,
                "start_index": parent_start + child["start_index"],
                "end_index": parent_start + child["end_index"],
            })

    return capped


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def chunk_file(
    source: str,
    rel_path: str,
    *,
    is_doc: bool = False,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[IndexChunk]:
    """
    Chunk a single file into ``IndexChunk`` objects.

    Uses Chonkie for Python and Markdown when available, and otherwise a
    ``RecursiveCharacterTextSplitter`` with language-aware separators where
    langchain provides them.
    """
    language = _detect_language(rel_path)

    raw_chunks: list[dict[str, Any]] = []
    if language in _CHONKIE_LANGUAGES:
        raw_chunks = _chonkie_chunk(source, language, chunk_size)

    if not raw_chunks:
        raw_chunks = _generic_chunk(source, language, chunk_size, chunk_overlap)
    else:
        raw_chunks = _split_oversized_chunks(raw_chunks, language, chunk_size, chunk_overlap)

    return [
        IndexChunk(
            text=rc["text"],
            file_path=rel_path,
            chunk_index=idx,
            content_hash=_content_hash(rc["text"]),
            language=language,
            is_doc=is_doc,
            start_index=rc.get("start_index", 0),
            end_index=rc.get("end_index", len(rc["text"])),
        )
        for idx, rc in enumerate(raw_chunks)
    ]
