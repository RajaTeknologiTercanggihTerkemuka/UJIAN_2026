-- Migration 003: Enable pgvector extension and HNSW cosine index for high-speed 1:N face search.
-- Prerequisites: pgvector extension must be installed on the PostgreSQL server
--   - Linux/VPS: apt install postgresql-<ver>-pgvector (or Docker image pgvector/pgvector:pg17)
--   - Run as a role with permission to CREATE EXTENSION vector (superuser / pg_extension owner).

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE face_templates
  ADD COLUMN IF NOT EXISTS embedding_vec vector(512);

-- Backfill existing JSON embeddings into native vector(512) column.
UPDATE face_templates
SET embedding_vec = (embedding::text)::vector(512)
WHERE embedding_vec IS NULL
  AND embedding IS NOT NULL;

-- HNSW index for fast approximate cosine similarity (<=> operator) on active templates.
CREATE INDEX IF NOT EXISTS ix_face_templates_embedding_hnsw
  ON face_templates
  USING hnsw (embedding_vec vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
