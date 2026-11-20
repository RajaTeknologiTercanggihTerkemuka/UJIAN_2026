from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(uri)
client = MlflowClient(tracking_uri=uri)
print("Tracking URI:", uri)
for exp in client.search_experiments():
    print(f"Experiment {exp.experiment_id}: {exp.name}")
print("Connection OK")
