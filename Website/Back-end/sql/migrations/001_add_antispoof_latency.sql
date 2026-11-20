-- Jalankan sekali pada database face_registration yang sudah memiliki tabel verification_events.
ALTER TABLE verification_events
ADD COLUMN IF NOT EXISTS antispoof_latency_ms DOUBLE PRECISION NULL;
