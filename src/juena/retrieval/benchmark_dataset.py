"""
Auto-generate retrieval benchmark queries from indexed repositories.

Extracts definitions from Python (via ``ast``), headings from Markdown, and
file-level identifiers from other languages.  Every generated entry is tagged
``label_source: auto`` so human curators can distinguish it from hand-written
gold queries.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import yaml

from juena.core.log import get_logger
from juena.indexing.repo_manager import RepoManager

logger = get_logger(__name__)

_GOLD_PATH_DEFAULT = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "retrieval" / "gold_queries.yaml"
)


@dataclass
class BenchmarkQuery:
    repo_id: str
    query: str
    relevant_files: List[str]
    label_source: str = "auto"
    query_type: str = "symbol"
    difficulty: str = "medium"
    notes: str = ""


# ------------------------------------------------------------------
# Extractors
# ------------------------------------------------------------------

def _python_definitions(source: str, rel_path: str, repo_id: str) -> list[BenchmarkQuery]:
    """Extract top-level functions and classes via the ``ast`` module."""
    queries: list[BenchmarkQuery] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return queries

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            queries.append(BenchmarkQuery(
                repo_id=repo_id,
                query=f"Where is the function {node.name} defined?",
                relevant_files=[rel_path],
                query_type="symbol",
                difficulty="easy",
            ))
        elif isinstance(node, ast.ClassDef):
            queries.append(BenchmarkQuery(
                repo_id=repo_id,
                query=f"What does class {node.name} do?",
                relevant_files=[rel_path],
                query_type="symbol",
                difficulty="medium",
            ))
    return queries


def _markdown_headings(source: str, rel_path: str, repo_id: str) -> list[BenchmarkQuery]:
    """Extract Markdown headings as documentation queries."""
    queries: list[BenchmarkQuery] = []
    for match in re.finditer(r"^#{1,3}\s+(.+)$", source, re.MULTILINE):
        heading = match.group(1).strip()
        if len(heading) < 4:
            continue
        queries.append(BenchmarkQuery(
            repo_id=repo_id,
            query=f"Where is the documentation about {heading}?",
            relevant_files=[rel_path],
            query_type="doc",
            difficulty="easy",
        ))
    return queries


def _file_level_query(rel_path: str, repo_id: str) -> BenchmarkQuery:
    """Create a simple file-discovery query from the path."""
    stem = Path(rel_path).stem
    return BenchmarkQuery(
        repo_id=repo_id,
        query=f"Where is the file {stem} located?",
        relevant_files=[rel_path],
        query_type="file",
        difficulty="easy",
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_queries(
    repo_manager: RepoManager,
    repo_ids: list[str] | None = None,
    *,
    max_per_repo: int = 50,
) -> list[BenchmarkQuery]:
    """
    Generate benchmark queries for the given repositories.

    Returns a flat list of ``BenchmarkQuery`` objects ready to be written to the
    gold-queries YAML.
    """
    target_ids = repo_ids or repo_manager.repo_ids
    all_queries: list[BenchmarkQuery] = []

    for repo_id in target_ids:
        repo_queries: list[BenchmarkQuery] = []
        files = repo_manager.list_files(repo_id)

        for rel_path in files:
            try:
                content = repo_manager.read_file(repo_id, rel_path)
            except Exception:
                continue

            lower = rel_path.lower()

            if lower.endswith(".py"):
                repo_queries.extend(_python_definitions(content, rel_path, repo_id))

            if lower.endswith((".md", ".rst")):
                repo_queries.extend(_markdown_headings(content, rel_path, repo_id))

            repo_queries.append(_file_level_query(rel_path, repo_id))

        if len(repo_queries) > max_per_repo:
            repo_queries = repo_queries[:max_per_repo]

        all_queries.extend(repo_queries)
        logger.info(
            "Generated %d benchmark queries for repo %s",
            len(repo_queries),
            repo_id,
        )

    return all_queries


def write_gold_yaml(
    queries: list[BenchmarkQuery],
    path: Path | None = None,
    *,
    merge: bool = True,
) -> Path:
    """
    Write benchmark queries to the gold-queries YAML file.

    When *merge* is ``True`` (default), existing ``human`` entries are preserved
    and only ``auto`` entries are replaced.
    """
    path = path or _GOLD_PATH_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_human: list[dict] = []
    if merge and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        existing_human = [
            q for q in data.get("queries", [])
            if q.get("label_source") == "human"
        ]

    merged = existing_human + [asdict(q) for q in queries]

    with open(path, "w") as f:
        f.write(
            "# Retrieval benchmark – gold queries\n"
            "#\n"
            "# Each entry pairs a natural-language query with the repository files that a\n"
            "# correct retrieval system should surface.  Entries generated automatically\n"
            "# carry ``label_source: auto``; colleagues can refine them or add new entries\n"
            "# with ``label_source: human``.\n"
            "#\n"
            "# Schema per entry\n"
            "# ----------------\n"
            "#   repo_id        – repository identifier from config/repositories.yaml\n"
            "#   query          – natural-language search query\n"
            "#   relevant_files – list of repo-relative file paths that should appear in results\n"
            "#   label_source   – \"auto\" | \"human\"\n"
            "#   query_type     – \"symbol\" | \"conceptual\" | \"file\" | \"doc\"\n"
            "#   difficulty     – \"easy\" | \"medium\" | \"hard\"  (optional)\n"
            "#   notes          – free-text annotation         (optional)\n\n"
        )
        yaml.dump({"queries": merged}, f, default_flow_style=False, sort_keys=False)

    logger.info("Wrote %d benchmark queries to %s", len(merged), path)
    return path
