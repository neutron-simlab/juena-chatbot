"""Bundled ingestion and database initialization contracts."""

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from juena.deployment import rag_ingest
from juena.deployment.rag_init import initialize


def test_ingest_forwards_options_and_reports_partial_failure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rag_ingest, "load_repositories", lambda _: {"first": None, "second": None})

    def ingest(repo_id, **options):
        calls.append((repo_id, options))
        return int(repo_id == "first")

    monkeypatch.setattr(rag_ingest, "ingest_repository", ingest)
    assert rag_ingest.main(["--limit", "2", "--dry-run", "--cache-dir", str(tmp_path)]) == 1
    assert [call[0] for call in calls] == ["first", "second"]
    assert calls[0][1]["limit"] == 2 and calls[0][1]["dry_run"] is True
    assert calls[0][1]["cache_dir"] == tmp_path
    with pytest.raises(SystemExit) as exc:
        rag_ingest.main(["unknown"])
    assert exc.value.code == 2
    assert len(calls) == 2


@pytest.fixture
def corpus_dsn():
    dsn = os.getenv("JUENA_TEST_RAG_DSN")
    if not dsn:
        pytest.skip("JUENA_TEST_RAG_DSN required for disposable pgvector databases")
    name = "juena_deploy_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(dsn, dbname=name, options="-c search_path=public")
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_initialization_preserves_populated_corpus_and_rejects_wrong_width(corpus_dsn):
    with psycopg.connect(corpus_dsn) as connection:
        initialize(connection, 8)
        connection.execute("INSERT INTO sources (source_id, category) VALUES ('test', 'code')")
        connection.execute("""
            INSERT INTO chunks (chunk_id, source_id, source_revision, chunk_index,
              category, text, content_hash, embedding, embedding_model, schema_version)
            VALUES ('chunk', 'test', 'revision', 0, 'code', 'preserve me', 'hash',
              '[1,1,1,1,1,1,1,1]', 'test-model', 1)
        """)
        connection.commit()
        initialize(connection, 8)
        with pytest.raises(RuntimeError, match="embedding width"):
            initialize(connection, 16)
        assert connection.execute("SELECT text FROM chunks").fetchall() == [("preserve me",)]


def test_partial_schema_is_not_silently_repaired(corpus_dsn):
    with psycopg.connect(corpus_dsn) as connection:
        connection.execute("CREATE TABLE sources (source_id text)")
        connection.commit()
        with pytest.raises(RuntimeError, match="Partial RAG schema"):
            initialize(connection, 8)
        assert connection.execute("SELECT to_regclass('chunks')").fetchone() == (None,)


def test_concurrent_initializers_serialize(corpus_dsn):
    def run():
        with psycopg.connect(corpus_dsn) as connection:
            initialize(connection, 8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in [executor.submit(run), executor.submit(run)]:
            future.result()
