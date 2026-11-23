#!/usr/bin/env bash
set -euo pipefail

echo "[TA v2] Menjalankan migrasi database & pengecekan pgvector HNSW..."
python "/app/Website/Back-end/scripts/apply_migrations.py"

echo "[TA v2] Menyinkronkan run V1 ke MLflow server jika tersedia..."
python "/app/deploy/seed_mlflow_vps.py" || true

echo "[TA v2] Memulai FastAPI server (1 worker untuk efisiensi RAM VPS)..."
exec uvicorn app.main:app --app-dir "/app/Website/Back-end" --host 0.0.0.0 --port 8000 --workers 1
