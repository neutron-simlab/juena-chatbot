"""
Chunking abstraction with Chonkie adapter and post-chunk enrichment.

Provides a normalized ``IndexChunk`` model that the rest of the codebase
depends on.  Chonkie is used as the parser-backed backend for Python and
Markdown; other languages fall back to the existing generic
``RecursiveCharacterTextSplitter``.
"""

from __future__ import annotations

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


# ------------------------------------------------------------------
# Enrichment
# ------------------------------------------------------------------

def _enrich_python_symbols(source: str, chunk_text: str, start: int) -> tuple[str, str]:
    """Attempt to identify the enclosing Python symbol for a chunk."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", ""

    best_name = ""
    best_type = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                node_start = source.index("\n") * 0  # approximate
                try:
                    func_source = ast.get_source_segment(source, node)
                    if func_source and chunk_text[:40] in func_source:
                        best_name = node.name
                        best_type = "function"
                except Exception:
                    pass
        elif isinstance(node, ast.ClassDef):
            try:
                cls_source = ast.get_source_segment(source, node)
                if cls_source and chunk_text[:40] in cls_source:
                    best_name = node.name
                    best_type = "class"
            except Exception:
                pass

    return best_name, best_type


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

    results: list[IndexChunk] = []
    for idx, rc in enumerate(raw_chunks):
        text = rc["text"]
        symbol = ""
        symbol_type = ""

        if language == "python":
            symbol, symbol_type = _enrich_python_symbols(source, text, rc.get("start_index", 0))

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
