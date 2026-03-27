"""
Persistent sparse lexical index backed by SQLite FTS5.

Provides ranked full-text search over indexed code and documentation chunks
without re-scanning every file at query time.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from juena.core.log import get_logger
from juena.indexing.repo_config import RepoConfig
from juena.indexing.repo_manager import RepoManager

logger = get_logger(__name__)

_SPARSE_DIR_ENV = "SPARSE_INDEX_DIR"
_DEFAULT_SPARSE_DIR = "data/sparse_index"


def _find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _sparse_dir() -> Path:
    env = os.getenv(_SPARSE_DIR_ENV)
    if env:
        return Path(env)
    return _find_workspace_root() / _DEFAULT_SPARSE_DIR


@dataclass
class SparseHit:
    file_path: str
    chunk_index: int
    content: str
    is_doc: bool
    bm25_rank: float


class RepoSparseIndex:
    """Manages a per-repo SQLite FTS5 index for ranked lexical retrieval."""

    def __init__(self, repo_manager: RepoManager) -> None:
        self._repo_manager = repo_manager
        self._connections: dict[str, sqlite3.Connection] = {}
        idx = _sparse_dir()
        idx.mkdir(parents=True, exist_ok=True)
        self._base_dir = idx

    def _db_path(self, repo_id: str) -> Path:
        safe = repo_id.replace("-", "_").replace(".", "_")[:50]
        return self._base_dir / f"sparse_{safe}.db"

    def _get_connection(self, repo_id: str) -> sqlite3.Connection:
        if repo_id not in self._connections:
            db = self._db_path(repo_id)
            conn = sqlite3.connect(str(db))
            conn.execute("PRAGMA journal_mode=WAL")
            self._connections[repo_id] = conn
        return self._connections[repo_id]

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                repo_id     TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                is_doc      INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL,
                content     TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TABLE IF NOT EXISTS index_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _is_doc_file(cfg: RepoConfig, rel_path: str) -> bool:
        for dp in cfg.docs_paths:
            if rel_path == dp or rel_path.startswith(dp.rstrip("/") + "/"):
                return True
        return False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        repo_id: str,
        chunks: list[dict[str, Any]],
        *,
        force: bool = False,
        repo_revision: str | None = None,
    ) -> int:
        """
        Build or rebuild the sparse FTS5 index for *repo_id*.

        *chunks* is a list of dicts with keys: file_path, chunk_index, is_doc,
        content_hash, content.

        Returns the number of rows indexed.
        """
        conn = self._get_connection(repo_id)
        self._ensure_tables(conn)

        if force:
            conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
            conn.commit()

        for chunk in chunks:
            chunk_id = f"{repo_id}::{chunk['file_path']}::{chunk['chunk_index']}::{chunk['content_hash']}"
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, repo_id, file_path, chunk_index, is_doc, content_hash, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    repo_id,
                    chunk["file_path"],
                    chunk["chunk_index"],
                    int(chunk.get("is_doc", False)),
                    chunk["content_hash"],
                    chunk["content"],
                ),
            )

        if repo_revision:
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                ("repo_revision", repo_revision),
            )

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]
        logger.info("Sparse-indexed %d chunks for repo %s", count, repo_id)
        return count

    def delete_file_chunks(self, repo_id: str, file_path: str) -> int:
        """Remove all chunks for a specific file. Returns rows deleted."""
        conn = self._get_connection(repo_id)
        self._ensure_tables(conn)
        cursor = conn.execute(
            "DELETE FROM chunks WHERE repo_id = ? AND file_path = ?",
            (repo_id, file_path),
        )
        conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Convert a natural-language query into valid FTS5 terms joined by OR."""
        tokens = re.findall(r"[a-zA-Z0-9_]+", query)
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "between", "through", "during",
            "and", "or", "but", "not", "no", "nor", "so", "yet",
            "if", "then", "than", "that", "this", "these", "those",
            "it", "its", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their",
            "what", "which", "who", "whom", "when", "where", "why", "how",
        }
        meaningful = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]
        if not meaningful:
            meaningful = tokens[:3]
        return " OR ".join(meaningful)

    def search(
        self,
        repo_id: str,
        query: str,
        *,
        max_results: int = 15,
        docs_only: bool = False,
    ) -> List[SparseHit]:
        """
        FTS5 ranked search against *repo_id*'s sparse index.

        Returns up to *max_results* ``SparseHit`` objects ordered by BM25 rank.
        """
        conn = self._get_connection(repo_id)
        self._ensure_tables(conn)

        fts_query = self._sanitize_fts5_query(query)
        if not fts_query:
            return []

        doc_filter = "AND c.is_doc = 1" if docs_only else ""
        sql = f"""
            SELECT c.file_path,
                   c.chunk_index,
                   c.content,
                   c.is_doc,
                   rank
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?
              AND c.repo_id = ?
              {doc_filter}
            ORDER BY rank
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, (fts_query, repo_id, max_results)).fetchall()
        except sqlite3.OperationalError:
            logger.debug("FTS5 query failed for %r – returning empty", fts_query)
            return []

        return [
            SparseHit(
                file_path=row[0],
                chunk_index=row[1],
                content=row[2],
                is_doc=bool(row[3]),
                bm25_rank=row[4],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def has_index(self, repo_id: str) -> bool:
        db = self._db_path(repo_id)
        if not db.exists():
            return False
        conn = self._get_connection(repo_id)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)
            ).fetchone()[0]
            return count > 0
        except sqlite3.OperationalError:
            return False

    def chunk_count(self, repo_id: str) -> int:
        conn = self._get_connection(repo_id)
        self._ensure_tables(conn)
        return conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]

    def close(self, repo_id: str | None = None) -> None:
        """Close database connections."""
        if repo_id:
            conn = self._connections.pop(repo_id, None)
            if conn:
                conn.close()
        else:
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()
