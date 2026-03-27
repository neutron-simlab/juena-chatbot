"""
Per-repo vector index backed by ChromaDB (persistent, on-disk).

Provides:
- build / incremental-update of an embeddings collection per repo
- semantic search returning ranked chunks with file + line metadata
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import DefaultEmbeddingFunction, Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from juena.core.log import get_logger
from juena.indexing.repo_config import RepoConfig
from juena.indexing.repo_manager import RepoManager

logger = get_logger(__name__)

_INDEX_DIR_ENV = "VECTOR_INDEX_DIR"
_DEFAULT_INDEX_DIR = "data/vector_index"
_PROGRESS_STEP_PERCENT = 5
_INDEX_SCHEMA_VERSION = 1
_COLLECTION_BASE_METADATA = {"hnsw:space": "cosine"}
_INDEX_META_SCHEMA_VERSION = "juena:index_schema_version"
_INDEX_META_REPO_REVISION = "juena:repo_revision"
_INDEX_META_FINGERPRINT = "juena:index_fingerprint"
_INDEX_META_EMBEDDING = "juena:embedding"


def _get_config():
    """Lazy import of Config to reuse the server's .env loading path."""
    from juena.core.config import Config
    return Config


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


def _should_log_progress(
    processed_files: int,
    total_files: int,
    next_progress_percent: int,
) -> tuple[bool, int]:
    """Return whether file-based indexing progress should be logged."""
    if total_files == 0:
        return False, next_progress_percent

    progress_percent = int(processed_files * 100 / total_files)
    if processed_files == total_files or progress_percent >= next_progress_percent:
        while next_progress_percent <= progress_percent:
            next_progress_percent += _PROGRESS_STEP_PERCENT
        return True, next_progress_percent

    return False, next_progress_percent


def _build_embedding_function() -> EmbeddingFunction[Documents]:
    config = _get_config()
    api_key = config.BLABLADOR_API_KEY
    base_url = config.BLABLADOR_BASE_URL
    model_name = config.BLABLADOR_EMBEDDING_MODEL

    if api_key and base_url:
        logger.info(
            "Using Blablador embedding model %s for repository indexing",
            model_name,
        )
        # Pass the API key directly because Chroma's OpenAI embedding function
        # prefers OPENAI_API_KEY when both OpenAI and Blablador credentials exist.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Direct api_key configuration will not be persisted.*",
                category=DeprecationWarning,
            )
            return OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=model_name,
            )

    logger.warning(
        "BLABLADOR_API_KEY or BLABLADOR_BASE_URL is not configured; "
        "falling back to Chroma's default embedding function for repository indexing.",
    )
    return DefaultEmbeddingFunction()


class RepoVectorIndex:
    """Manages a ChromaDB collection per repository."""

    def __init__(self, repo_manager: RepoManager) -> None:
        self._repo_manager = repo_manager
        idx = _index_dir()
        idx.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(idx))
        self._collections: dict[str, chromadb.Collection] = {}
        self._embedding_function = _build_embedding_function()
        self._embedding_descriptor = self._build_embedding_descriptor()

    def _collection_name(self, repo_id: str) -> str:
        safe = repo_id.replace("-", "_").replace(".", "_")[:50]
        return f"repo_{safe}"

    def _build_embedding_descriptor(self) -> str:
        config = _get_config()
        if config.BLABLADOR_API_KEY and config.BLABLADOR_BASE_URL:
            return f"blablador:{config.BLABLADOR_BASE_URL}:{config.BLABLADOR_EMBEDDING_MODEL}"
        return "chroma-default"

    def _index_fingerprint(self, repo_id: str) -> str:
        cfg = self._repo_manager.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        payload = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "embedding": self._embedding_descriptor,
            "include_globs": cfg.include_globs,
            "exclude_globs": cfg.exclude_globs,
            "max_file_bytes": cfg.max_file_bytes,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "docs_paths": cfg.docs_paths,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def _index_metadata(self, repo_id: str, repo_revision: str) -> dict[str, Any]:
        return {
            _INDEX_META_SCHEMA_VERSION: _INDEX_SCHEMA_VERSION,
            _INDEX_META_REPO_REVISION: repo_revision,
            _INDEX_META_FINGERPRINT: self._index_fingerprint(repo_id),
            _INDEX_META_EMBEDDING: self._embedding_descriptor,
        }

    def _collection_creation_metadata(self, repo_id: str, repo_revision: str) -> dict[str, Any]:
        return {
            **_COLLECTION_BASE_METADATA,
            **self._index_metadata(repo_id, repo_revision),
        }

    def _get_or_create_collection(
        self,
        repo_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> chromadb.Collection:
        if repo_id not in self._collections:
            name = self._collection_name(repo_id)
            self._collections[repo_id] = self._client.get_or_create_collection(
                name=name,
                metadata=metadata or dict(_COLLECTION_BASE_METADATA),
                embedding_function=self._embedding_function,
            )
        return self._collections[repo_id]

    def _get_collection_if_exists(self, repo_id: str) -> chromadb.Collection | None:
        if not self.has_collection(repo_id):
            return None
        if repo_id not in self._collections:
            self._collections[repo_id] = self._client.get_collection(
                name=self._collection_name(repo_id),
                embedding_function=self._embedding_function,
            )
        return self._collections[repo_id]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        repo_id: str,
        *,
        force: bool = False,
        repo_revision: str | None = None,
        collect_chunks: bool = False,
    ) -> int | tuple[int, list[dict[str, Any]]]:
        """
        Build (or rebuild) the vector index for *repo_id*.

        When *collect_chunks* is ``True``, returns ``(count, chunks)`` where
        *chunks* is a list of dicts suitable for feeding into the sparse index.
        Otherwise returns just the chunk count.
        """
        cfg = self._repo_manager.get_config(repo_id)
        if cfg is None:
            raise ValueError(f"Unknown repo: {repo_id}")

        repo_revision = repo_revision or self._repo_manager.current_revision(repo_id)
        index_metadata = self._index_metadata(repo_id, repo_revision)
        creation_metadata = self._collection_creation_metadata(repo_id, repo_revision)

        if force:
            if self.has_collection(repo_id):
                self._client.delete_collection(self._collection_name(repo_id))
            self._collections.pop(repo_id, None)

        col = self._get_or_create_collection(repo_id, metadata=creation_metadata)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            length_function=len,
        )

        files = self._repo_manager.list_files(repo_id)
        total_files = len(files)
        total_chunks = 0
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[dict[str, Any]] = []
        collected: list[dict[str, Any]] = [] if collect_chunks else []
        BATCH_SIZE = 200
        next_progress_percent = _PROGRESS_STEP_PERCENT

        if total_files == 0:
            logger.info(
                "Indexing progress for repo %s: 100%% (0/0 files, 0 chunks)",
                repo_id,
            )
            return (0, []) if collect_chunks else 0

        logger.info(
            "Indexing progress for repo %s: 0%% (0/%d files, 0 chunks)",
            repo_id,
            total_files,
        )

        for processed_files, rel_path in enumerate(files, start=1):
            try:
                content = self._repo_manager.read_file(repo_id, rel_path)
            except Exception as exc:
                logger.debug("Skipping %s/%s: %s", repo_id, rel_path, exc)
            else:
                is_doc = self._is_doc_file(cfg, rel_path)
                chunks = splitter.split_text(content)

                for idx, chunk in enumerate(chunks):
                    c_hash = _content_hash(chunk)
                    chunk_id = f"{repo_id}::{rel_path}::{idx}::{c_hash}"
                    meta = {
                        "repo_id": repo_id,
                        "file_path": rel_path,
                        "chunk_index": idx,
                        "is_doc": is_doc,
                        "content_hash": c_hash,
                    }
                    batch_ids.append(chunk_id)
                    batch_docs.append(chunk)
                    batch_metas.append(meta)
                    total_chunks += 1

                    if collect_chunks:
                        collected.append({**meta, "content": chunk})

                    if len(batch_ids) >= BATCH_SIZE:
                        col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                        batch_ids, batch_docs, batch_metas = [], [], []

            should_log, next_progress_percent = _should_log_progress(
                processed_files,
                total_files,
                next_progress_percent,
            )
            if should_log:
                progress_percent = int(processed_files * 100 / total_files)
                logger.info(
                    "Indexing progress for repo %s: %d%% (%d/%d files, %d chunks)",
                    repo_id,
                    progress_percent,
                    processed_files,
                    total_files,
                    total_chunks,
                )

        if batch_ids:
            col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

        col.modify(metadata=index_metadata)
        logger.info("Indexed %d chunks for repo %s", total_chunks, repo_id)
        if collect_chunks:
            return total_chunks, collected
        return total_chunks

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(
        self,
        repo_id: str,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
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

    def has_collection(self, repo_id: str) -> bool:
        name = self._collection_name(repo_id)
        for collection in self._client.list_collections():
            collection_name = collection if isinstance(collection, str) else getattr(collection, "name", None)
            if collection_name == name:
                return True
        return False

    def collection_count_existing(self, repo_id: str) -> int | None:
        col = self._get_collection_if_exists(repo_id)
        if col is None:
            return None
        return col.count()

    def delete_file_chunks(self, repo_id: str, file_path: str) -> int:
        """Remove all chunks for a specific file from the collection."""
        col = self._get_collection_if_exists(repo_id)
        if col is None:
            return 0
        existing = col.get(where={"file_path": file_path})
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def upsert_file_chunks(
        self,
        repo_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Upsert chunks for a single file into the collection."""
        col = self._get_or_create_collection(repo_id)
        ids = []
        docs = []
        metas = []
        for chunk in chunks:
            c_hash = chunk["content_hash"]
            chunk_id = f"{repo_id}::{chunk['file_path']}::{chunk['chunk_index']}::{c_hash}"
            ids.append(chunk_id)
            docs.append(chunk["content"])
            metas.append({
                "repo_id": repo_id,
                "file_path": chunk["file_path"],
                "chunk_index": chunk["chunk_index"],
                "is_doc": chunk.get("is_doc", False),
                "content_hash": c_hash,
            })
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def index_staleness_reason(self, repo_id: str, repo_revision: str) -> str | None:
        col = self._get_collection_if_exists(repo_id)
        if col is None:
            return "vector index missing"

        if col.count() == 0:
            return "vector index empty"

        metadata = col.metadata or {}
        if metadata.get(_INDEX_META_SCHEMA_VERSION) != _INDEX_SCHEMA_VERSION:
            return "index metadata missing or outdated"

        if metadata.get(_INDEX_META_REPO_REVISION) != repo_revision:
            return "repository revision changed"

        if metadata.get(_INDEX_META_FINGERPRINT) != self._index_fingerprint(repo_id):
            return "index configuration changed"

        if metadata.get(_INDEX_META_EMBEDDING) != self._embedding_descriptor:
            return "embedding configuration changed"

        return None
