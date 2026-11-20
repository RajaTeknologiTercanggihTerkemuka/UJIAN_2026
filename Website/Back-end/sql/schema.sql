-- Run in database face_registration after connecting as face_app_user or database owner.
CREATE TABLE IF NOT EXISTS users (
  id BIGSERIAL PRIMARY KEY,
  nim VARCHAR(80) UNIQUE NOT NULL,
  full_name VARCHAR(200) NOT NULL,
  status VARCHAR(30) NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS face_templates (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  embedding JSON NOT NULL,
  model_version VARCHAR(100) NOT NULL,
  preprocessing_version VARCHAR(100) NOT NULL,
  source_type VARCHAR(50) NOT NULL DEFAULT 'admin_reference_photo',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_face_template_version ON face_templates(model_version, preprocessing_version);
CREATE INDEX IF NOT EXISTS ix_face_templates_user_id ON face_templates(user_id);
CREATE TABLE IF NOT EXISTS verification_events (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
  nim_masked VARCHAR(40),
  mode VARCHAR(10) NOT NULL DEFAULT '1:1',
  face_detected BOOLEAN NOT NULL DEFAULT FALSE,
  face_count INTEGER NOT NULL DEFAULT 0,
  detection_confidence DOUBLE PRECISION,
  face_box_area_ratio DOUBLE PRECISION,
  similarity_score DOUBLE PRECISION,
  threshold_used DOUBLE PRECISION,
  decision_status VARCHAR(40) NOT NULL,
  failure_reason TEXT,
  quality_warnings JSON NOT NULL DEFAULT '[]',
  quality_metrics JSON NOT NULL DEFAULT '{}',
  antispoof_status VARCHAR(40) NOT NULL DEFAULT 'not_configured',
  antispoof_score DOUBLE PRECISION,
  detection_latency_ms DOUBLE PRECISION,
  embedding_latency_ms DOUBLE PRECISION,
  comparison_latency_ms DOUBLE PRECISION,
  total_latency_ms DOUBLE PRECISION,
  model_version VARCHAR(100),
  preprocessing_version VARCHAR(100),
  threshold_version VARCHAR(100),
  request_id VARCHAR(80),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_verification_events_created_at ON verification_events(created_at DESC);
CREATE INDEX IF NOT EXISTS ix_verification_events_user_id ON verification_events(user_id);

ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS antispoof_latency_ms DOUBLE PRECISION NULL;
