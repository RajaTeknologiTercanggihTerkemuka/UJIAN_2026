from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

backend_uri = os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
if not backend_uri:
    # Use PostgreSQL when configured, otherwise safe local SQLite for a first UI smoke test.
    backend_uri = os.getenv("MLFLOW_SQLITE_URI", f"sqlite:///{(ROOT / 'MLOps' / 'mlflow' / 'mlflow.db').as_posix()}")

artifact_dir = ROOT / "MLOps" / "mlflow" / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
artifact_uri = artifact_dir.resolve().as_uri()

cmd = [
    sys.executable, "-m", "mlflow", "server",
    "--backend-store-uri", backend_uri,
    "--default-artifact-root", artifact_uri,
    "--host", "127.0.0.1",
    "--port", os.getenv("MLFLOW_PORT", "5000"),
]
print("Starting MLflow on http://127.0.0.1:5000")
print("Backend store:", "PostgreSQL" if "postgresql" in backend_uri else "local SQLite")
print("Artifact directory:", artifact_dir)
print("Command:", " ".join("<redacted>" if "password" in x.lower() else x for x in cmd))
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
