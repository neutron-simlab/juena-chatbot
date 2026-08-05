"""
Chunking abstraction with Chonkie adapter and post-chunk enrichment.

Provides a normalized ``IndexChunk`` model that the rest of the codebase
depends on.  Chonkie is used as the parser-backed backend for Python and
Markdown; other languages fall back to the existing generic
``RecursiveCharacterTextSplitter``.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from juena.core.log import get_logger

logger = get_logger(__name__)

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "markdown",
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


@dataclass
class IndexChunk:
    """Normalized chunk emitted by the chunking layer."""

    text: str
    file_path: str
    chunk_index: int
    content_hash: str
    language: str
    is_doc: bool
    start_index: int = 0
    end_index: int = 0
    symbol: str = ""
    symbol_type: str = ""
    path_tokens: List[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _path_tokens(rel_path: str) -> list[str]:
    parts = Path(rel_path).parts
    stem = Path(rel_path).stem
    tokens = list(parts[:-1]) + [stem]
    expanded: list[str] = []
    for t in tokens:
        expanded.extend(re.split(r"[_\-./]", t))
    return [tok.lower() for tok in expanded if tok]


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

def _generic_chunk(
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(source)
    result: list[dict[str, Any]] = []
    offset = 0
    for t in texts:
        idx = source.find(t, offset)
        if idx == -1:
            idx = offset
        result.append({
            "text": t,
            "start_index": idx,
            "end_index": idx + len(t),
            "token_count": len(t),
        })
        offset = idx + 1
    return result


def _split_oversized_chunks(
    chunks: list[dict[str, Any]],
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
        for child in _generic_chunk(text, chunk_size, safe_overlap):
            start_index = parent_start + child["start_index"]
            capped.append({
                **child,
                "start_index": start_index,
                "end_index": parent_start + child["end_index"],
            })

    return capped


# ------------------------------------------------------------------
# Enrichment
# ------------------------------------------------------------------

def _build_python_symbol_index(source: str) -> list[tuple[int, int, str, str]]:
    """Parse the file once and index (start_line, end_line, name, type) for
    every function/class definition, keyed by 1-indexed source line."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    index: list[tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            index.append((node.lineno, end_lineno, node.name, kind))

    return index


def _line_start_offsets(source: str) -> list[int]:
    """offsets[i] is the character offset where line i+1 (1-indexed) starts."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _symbol_at_offset(
    symbol_index: list[tuple[int, int, str, str]],
    line_offsets: list[int],
    start: int,
) -> tuple[str, str]:
    """Find the innermost function/class enclosing a character offset."""
    if not symbol_index:
        return "", ""

    line = bisect.bisect_right(line_offsets, start)
    best: tuple[int, int, str, str] | None = None
    for sym in symbol_index:
        start_line, end_line, _, _ = sym
        if start_line <= line <= end_line and (
            best is None or (end_line - start_line) < (best[1] - best[0])
        ):
            best = sym

    return (best[2], best[3]) if best else ("", "")


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
    Chunk a single file into enriched ``IndexChunk`` objects.

    Uses Chonkie for Python and Markdown when available, falls back to
    the generic ``RecursiveCharacterTextSplitter`` for other languages.
    """
    language = _detect_language(rel_path)
    ptokens = _path_tokens(rel_path)

    raw_chunks: list[dict[str, Any]] = []
    if language in _CHONKIE_LANGUAGES:
        raw_chunks = _chonkie_chunk(source, language, chunk_size)

    if not raw_chunks:
        raw_chunks = _generic_chunk(source, chunk_size, chunk_overlap)
    else:
        raw_chunks = _split_oversized_chunks(raw_chunks, chunk_size, chunk_overlap)

    symbol_index: list[tuple[int, int, str, str]] = []
    line_offsets: list[int] = []
    if language == "python":
        symbol_index = _build_python_symbol_index(source)
        line_offsets = _line_start_offsets(source)

    results: list[IndexChunk] = []
    for idx, rc in enumerate(raw_chunks):
        text = rc["text"]
        symbol = ""
        symbol_type = ""

        if language == "python":
            symbol, symbol_type = _symbol_at_offset(symbol_index, line_offsets, rc.get("start_index", 0))

        results.append(IndexChunk(
            text=text,
            file_path=rel_path,
            chunk_index=idx,
            content_hash=_content_hash(text),
            language=language,
            is_doc=is_doc,
            start_index=rc.get("start_index", 0),
            end_index=rc.get("end_index", len(text)),
            symbol=symbol,
            symbol_type=symbol_type,
            path_tokens=ptokens,
        ))

    return results
