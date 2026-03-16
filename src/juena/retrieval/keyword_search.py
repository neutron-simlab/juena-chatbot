"""
Keyword (grep-style) search across repo files.

Provides ranked snippet results with surrounding context lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from juena.core.log import get_logger
from juena.retrieval.repo_manager import RepoManager

logger = get_logger(__name__)

CONTEXT_LINES = 4  # lines of context above and below a match


@dataclass
class KeywordHit:
    file_path: str
    line_number: int
    snippet: str
    match_count: int
    is_doc: bool


def keyword_search(
    repo_manager: RepoManager,
    repo_id: str,
    query: str,
    *,
    max_hits: int = 15,
    case_sensitive: bool = False,
) -> List[KeywordHit]:
    """
    Search files in *repo_id* for lines matching *query*.

    Returns up to *max_hits* ``KeywordHit`` objects sorted by match count
    (descending), then file path.
    """
    cfg = repo_manager.get_config(repo_id)
    if cfg is None:
        raise ValueError(f"Unknown repo: {repo_id}")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(re.escape(query), flags)
    except re.error:
        pattern = re.compile(re.escape(query), flags)

    files = repo_manager.list_files(repo_id)
    all_hits: list[KeywordHit] = []

    doc_paths = cfg.docs_paths

    for rel_path in files:
        try:
            content = repo_manager.read_file(repo_id, rel_path)
        except Exception:
            continue

        lines = content.splitlines()
        is_doc = _is_doc(rel_path, doc_paths)

        file_matches: list[int] = []
        for i, line in enumerate(lines):
            if pattern.search(line):
                file_matches.append(i)

        if not file_matches:
            continue

        for line_idx in file_matches:
            start = max(0, line_idx - CONTEXT_LINES)
            end = min(len(lines), line_idx + CONTEXT_LINES + 1)
            snippet_lines = []
            for j in range(start, end):
                prefix = ">>>" if j == line_idx else "   "
                snippet_lines.append(f"{prefix} {j + 1:>5}| {lines[j]}")
            snippet = "\n".join(snippet_lines)

            all_hits.append(KeywordHit(
                file_path=rel_path,
                line_number=line_idx + 1,
                snippet=snippet,
                match_count=len(file_matches),
                is_doc=is_doc,
            ))

    all_hits.sort(key=lambda h: (-h.match_count, h.file_path, h.line_number))

    # Deduplicate overlapping snippets from the same file
    seen: set[tuple[str, int]] = set()
    deduped: list[KeywordHit] = []
    for h in all_hits:
        key = (h.file_path, h.line_number)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
        if len(deduped) >= max_hits:
            break

    return deduped


def _is_doc(rel_path: str, doc_paths: list[str]) -> bool:
    for dp in doc_paths:
        if rel_path == dp or rel_path.startswith(dp.rstrip("/") + "/"):
            return True
    return False
