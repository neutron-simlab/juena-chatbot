"""
Per-repo vector index backed by ChromaDB (persistent, on-disk).

Provides:
- build / incremental-update of an embeddings collection per repo
- semantic search returning ranked chunks with file + line metadata
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from juena.core.log import get_logger
from juena.retrieval.repo_config import RepoConfig
from juena.retrieval.repo_manager import RepoManager

logger = get_logger(__name__)

_INDEX_DIR_ENV = "VECTOR_INDEX_DIR"
_DEFAULT_INDEX_DIR = "data/vector_index"


def _find_workspace_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _index_dir() -> Path:
    env = os.getenv(_INDEX_DIR_ENV)
    if env:
        return Path(env)
    return _find_workspace_root() / _DEFAULT_INDEX_DIR


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class RepoVectorIndex:
    """Manages a ChromaDB collection per repository."""

    def __init__(self, repo_manager: RepoManager) -> None:
        self._repo_manager = repo_manager
        idx = _index_dir()
        idx.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(idx))
        self._collections: Dict[str, chromadb.Collection] = {}

    def _collection_name(self, repo_id: str) -> str:
        safe = repo_id.replace("-", "_").replace(".", "_")[:50]
        return f"repo_{safe}"

    def _get_or_create_collection(self, repo_id: str) -> chromadb.Collection:
        if repo_id not in self._collections:
            name = self._collection_name(repo_id)
            self._collections[repo_id] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[repo_id]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, repo_id: str, *, force: bool = False) -> int:
        """
        Build (or rebuild) the vector index for *repo_id*.

        Returns the number of chunks indexed.
        """
        cfg = self._repo_manager.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        col = self._get_or_create_collection(repo_id)

        if force:
            self._client.delete_collection(self._collection_name(repo_id))
            self._collections.pop(repo_id, None)
            col = self._get_or_create_collection(repo_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            length_function=len,
        )

        files = self._repo_manager.list_files(repo_id)
        total_chunks = 0
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict[str, Any]] = []
        BATCH_SIZE = 200

        for rel_path in files:
            try:
                content = self._repo_manager.read_file(repo_id, rel_path)
            except Exception as exc:
                logger.debug("Skipping %s/%s: %s", repo_id, rel_path, exc)
                continue

            is_doc = self._is_doc_file(cfg, rel_path)
            chunks = splitter.split_text(content)

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{repo_id}::{rel_path}::{idx}::{_content_hash(chunk)}"
                batch_ids.append(chunk_id)
                batch_docs.append(chunk)
                batch_metas.append({
                    "repo_id": repo_id,
                    "file_path": rel_path,
                    "chunk_index": idx,
                    "is_doc": is_doc,
                    "content_hash": _content_hash(chunk),
                })
                total_chunks += 1

                if len(batch_ids) >= BATCH_SIZE:
                    col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                    batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

        logger.info("Indexed %d chunks for repo %s", total_chunks, repo_id)
        return total_chunks

    def build_all(self, *, force: bool = False) -> Dict[str, int]:
        """Build indices for every configured repo. Returns {repo_id: chunk_count}."""
        results: Dict[str, int] = {}
        for repo_id in self._repo_manager.repo_ids:
            try:
                results[repo_id] = self.build_index(repo_id, force=force)
            except Exception as exc:
                logger.error("Failed to index repo %s: %s", repo_id, exc)
                results[repo_id] = 0
        return results

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(
        self,
        repo_id: str,
        query: str,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search against *repo_id*'s collection.

        Returns a list of dicts with keys:
            file_path, chunk_index, is_doc, distance, content
        """
        col = self._get_or_create_collection(repo_id)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, col.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            hits.append({
                "id": doc_id,
                "file_path": metas[i].get("file_path", ""),
                "chunk_index": metas[i].get("chunk_index", 0),
                "is_doc": metas[i].get("is_doc", False),
                "distance": dists[i] if i < len(dists) else None,
                "content": docs[i] if i < len(docs) else "",
            })
        return hits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_doc_file(cfg: RepoConfig, rel_path: str) -> bool:
        for dp in cfg.docs_paths:
            if rel_path == dp or rel_path.startswith(dp.rstrip("/") + "/"):
                return True
        return False

    def collection_count(self, repo_id: str) -> int:
        col = self._get_or_create_collection(repo_id)
        return col.count()
