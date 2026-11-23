-- Migration 002: Anti-Joki (1:N deduplication), participant exam metadata, and duplicate identity alerts.
-- Safe and idempotent on existing databases containing enrolled users/templates.

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS exam_year INTEGER NOT NULL DEFAULT 2026;

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE face_templates
  ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

CREATE INDEX IF NOT EXISTS ix_face_templates_active_ver
  ON face_templates (model_version, preprocessing_version, is_active);

ALTER TABLE verification_events
  ALTER COLUMN mode TYPE VARCHAR(30);

ALTER TABLE verification_events
  ADD COLUMN IF NOT EXISTS matched_user_id BIGINT REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE verification_events
  ADD COLUMN IF NOT EXISTS matched_nim_masked VARCHAR(40);

ALTER TABLE verification_events
  ADD COLUMN IF NOT EXISTS search_backend VARCHAR(40);

ALTER TABLE verification_events
  ADD COLUMN IF NOT EXISTS candidate_count INTEGER;

CREATE TABLE IF NOT EXISTS duplicate_identity_alerts (
  id BIGSERIAL PRIMARY KEY,
  event_id BIGINT REFERENCES verification_events(id) ON DELETE SET NULL,
  claimed_nim VARCHAR(80) NOT NULL,
  claimed_nim_masked VARCHAR(40) NOT NULL,
  claimed_name VARCHAR(200),
  matched_user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
  matched_nim_masked VARCHAR(40),
  similarity_score DOUBLE PRECISION,
  threshold_used DOUBLE PRECISION,
  alert_type VARCHAR(50) NOT NULL,
  review_status VARCHAR(30) NOT NULL DEFAULT 'pending',
  reviewer_notes TEXT,
  model_version VARCHAR(100),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_dup_alerts_created_at
  ON duplicate_identity_alerts (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_dup_alerts_review_status
  ON duplicate_identity_alerts (review_status);
