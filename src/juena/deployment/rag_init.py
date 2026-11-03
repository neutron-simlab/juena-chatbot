"""Create a fresh corpus once; validate existing storage without migrating it."""

import os

import psycopg
from juena_rag.store.schema import apply_ddl


def initialize(connection, dimension: int) -> None:
    with connection.transaction():
        # Concurrent Compose invocations must not both observe an empty schema.
        connection.execute("SELECT pg_advisory_xact_lock(736482109)")
        tables = connection.execute(
            "SELECT to_regclass('public.sources'), to_regclass('public.chunks')"
        ).fetchone()
        if tables == (None, None):
            apply_ddl(connection, dimension=dimension, create_extension=True)
        elif not all(tables):
            raise RuntimeError("Partial RAG schema; restore or migrate it explicitly.")

        # These are the columns consumed by the pinned service and ingestion API.
        # A table's name alone is insufficient evidence of a compatible schema.
        connection.execute(
            "SELECT source_id, category, title, active_revision, artifact_uri, "
            "artifact_format, artifact_hash, parser, updated_at FROM sources LIMIT 0"
        )
        connection.execute(
            "SELECT chunk_id, source_id, source_revision, chunk_index, category, "
            "language, chunk_kind, source_uri, text, content_hash, extra, embedding, "
            "embedding_model, schema_version, ingested_at FROM chunks LIMIT 0"
        )
        width = connection.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'public.chunks'::regclass AND attname = 'embedding' "
            "AND atttypid = 'vector'::regtype"
        ).fetchone()
        if width != (dimension,):
            raise RuntimeError(f"RAG embedding width must be {dimension}; refusing to alter existing data.")
        indexes = connection.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
            "AND tablename IN ('sources', 'chunks')"
        ).fetchall()
        required = {
            "sources_pkey", "chunks_pkey", "chunks_embedding_bits_idx",
            "chunks_revision_idx", "chunks_category_idx", "chunks_kind_idx", "chunks_extra_idx",
        }
        if missing := required - {row[0] for row in indexes}:
            raise RuntimeError(f"Incomplete RAG schema: missing indexes {sorted(missing)}")


def main() -> None:
    dimension = int(os.environ.get("BLABLADOR_EMBEDDING_DIMENSION", "4096"))
    with psycopg.connect(os.environ["JUENA_RAG_DSN"], options="-c search_path=public") as connection:
        initialize(connection, dimension)
    print(f"RAG schema ready (dimension={dimension}); existing corpus preserved.")


if __name__ == "__main__":
    main()
