"""
Seed real V1 Partial Fine-Tuning (IMSFD) experiment, 15-epoch metrics, test metrics, and PAD calibration
into the VPS MLflow Tracking Server via MLflow REST API (zero extra Python deps required).
Idempotent: skips if a run already exists in TA-v2-Face-Recognition.
"""
from __future__ import annotations
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
EXP_NAME = "TA-v2-Face-Recognition"


def api(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{TRACKING_URI}/api/2.0/mlflow/{path.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_mlflow() -> bool:
    for _ in range(20):
        try:
            with urllib.request.urlopen(f"{TRACKING_URI}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    if not wait_mlflow():
        print(f"[MLFLOW SEED] MLflow belum siap di {TRACKING_URI}, lewati seed.")
        return 0

    # Get or create experiment
    try:
        exp = api("GET", f"experiments/get-by-name?experiment_name={EXP_NAME}")["experiment"]
        exp_id = exp["experiment_id"]
    except urllib.error.HTTPError:
        exp_id = api("POST", "experiments/create", {"name": EXP_NAME})["experiment_id"]

    # Check if run already exists
    existing = api("POST", "runs/search", {"experiment_ids": [exp_id], "max_results": 1})
    if existing.get("runs"):
        print(f"[MLFLOW SEED] Experiment {EXP_NAME} sudah memiliki run. Skip.")
        return 0

    now_ms = int(time.time() * 1000)
    run = api("POST", "runs/create", {
        "experiment_id": exp_id,
        "run_name": "imsfd_partial_ft_v1",
        "start_time": now_ms,
        "tags": [
            {"key": "mlflow.runName", "value": "imsfd_partial_ft_v1"},
            {"key": "import_type", "value": "retrospective_colab_v1"},
            {"key": "dataset", "value": "IMSFD_68_identities"},
            {"key": "anti_spoof_model", "value": "2.7_80x80_MiniFASNetV2"},
            {"key": "vector_search", "value": "pgvector_hnsw"},
        ],
    })["run"]
    run_id = run["info"]["run_id"]

    params = [
        {"key": "architecture", "value": "InceptionResnetV1"},
        {"key": "pretrained", "value": "vggface2"},
        {"key": "adaptation", "value": "partial_fine_tuning"},
        {"key": "unfrozen_modules", "value": "repeat_3,block8,last_linear,last_bn"},
        {"key": "objective", "value": "ArcFace + CrossEntropyLoss"},
        {"key": "arcface_scale", "value": "64.0"},
        {"key": "arcface_margin", "value": "0.50"},
        {"key": "embedding_dim", "value": "512"},
        {"key": "epochs", "value": "15"},
        {"key": "batch_size", "value": "16"},
        {"key": "verify_threshold_1to1", "value": "0.3651"},
        {"key": "dedup_threshold_1toN", "value": "0.75"},
        {"key": "pad_live_threshold", "value": "0.4302"},
    ]

    metrics = []
    hist_path = PROJECT_ROOT / "MLOps" / "imports" / "imsfd_partial_ft_v1" / "training_history.csv"
    if hist_path.exists():
        with hist_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                step = int(float(row["epoch"]))
                ts = now_ms + step * 1000
                for key, col in [
                    ("train_loss", "train_loss"),
                    ("val_loss", "val_loss"),
                    ("train_arcface_top1", "train_arcface_top1"),
                    ("val_arcface_top1", "val_arcface_top1"),
                ]:
                    metrics.append({"key": key, "value": float(row[col]), "timestamp": ts, "step": step})

    tm_path = PROJECT_ROOT / "MLOps" / "imports" / "imsfd_partial_ft_v1" / "verification_metrics_test.json"
    if tm_path.exists():
        tm = json.loads(tm_path.read_text(encoding="utf-8"))
        for k in ["test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_FAR", "test_FRR", "test_ROC_AUC"]:
            if k in tm:
                metrics.append({"key": k, "value": float(tm[k]), "timestamp": now_ms, "step": 15})

    pad_path = PROJECT_ROOT / "Face Recognition" / "artifacts" / "thresholds" / "pad_threshold_v1.json"
    if pad_path.exists():
        pt = json.loads(pad_path.read_text(encoding="utf-8")).get("test_metrics_at_frozen_validation_threshold", {})
        for k, mk in [("ROC_AUC", "pad_test_ROC_AUC"), ("BPCER", "pad_test_BPCER"), ("APCER", "pad_test_APCER"), ("ACER", "pad_test_ACER")]:
            if k in pt:
                metrics.append({"key": mk, "value": float(pt[k]), "timestamp": now_ms, "step": 15})

    api("POST", "runs/log-batch", {"run_id": run_id, "params": params, "metrics": metrics})
    api("POST", "runs/update", {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)})
    print(f"[MLFLOW SEED] Berhasil mengisi run V1 ({len(params)} params, {len(metrics)} metrics) ke {TRACKING_URI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
