"""Retrospectively import an already-completed IMSFD V1 run into MLflow.

Does not retrain. Logs epoch history and selected artifacts. Never uploads face
images, identity mappings, NIM, or private masterdata.
"""
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
from typing import Any
import mlflow

def numeric(value: Any):
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def log_history(path: Path):
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return
    count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    aliases = {
        "train_loss": ["train_loss"],
        "validation_loss": ["val_loss", "validation_loss"],
        "train_top1_accuracy": ["train_arcface_top1", "train_acc", "train_accuracy"],
        "validation_top1_accuracy": ["val_arcface_top1", "val_acc", "val_accuracy"],
        "backbone_learning_rate": ["backbone_lr"],
        "arcface_learning_rate": ["arcface_lr"],
        "epoch_seconds": ["seconds"],
    }
    for i, row in enumerate(rows, start=1):
        step = int(numeric(row.get("epoch")) or i)
        for metric, candidates in aliases.items():
            value = next((numeric(row.get(k)) for k in candidates if numeric(row.get(k)) is not None), None)
            if value is not None:
                mlflow.log_metric(metric, value, step=step)
                count += 1
    print(f"[OK] Backfilled {count} epoch metrics from {len(rows)} rows")

def log_final_metrics(path: Path):
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for key, value in data.items():
        number = numeric(value)
        if number is not None:
            mlflow.log_metric(key.lower().replace(" ", "_").replace("-", "_")[:250], number)
            count += 1
    print(f"[OK] Logged {count} final metrics")

def log_file(path: Path, artifact_path: str):
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        print("[ARTIFACT]", path.name)

def log_dir(path: Path, artifact_path: str):
    if path.is_dir() and any(p.is_file() for p in path.rglob("*")):
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        print("[ARTIFACT DIR]", path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--notebook", default=None)
    parser.add_argument("--experiment", default="TA-v2-Face-Recognition")
    parser.add_argument("--run-name", default="IMSFD_Partial_FT_V1_Historical_Import")
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    if not out.exists():
        raise FileNotFoundError(out)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    params = {
        "architecture": "InceptionResnetV1",
        "initial_pretrained_weights": "VGGFace2",
        "target_dataset": "IMSFD",
        "objective": "ArcFace angular-margin head + CrossEntropyLoss",
        "input_size": "160x160",
        "embedding_dim": "512",
        "preprocessing_version": "squarepad_resize160_rgb_standardization_v1",
        "mtcnn_used_during_training": "false",
        "unfrozen_modules": "repeat_3,block8,last_linear,last_bn",
        "arcface_margin": 0.50,
        "arcface_scale": 64.0,
        "backbone_learning_rate": 1e-5,
        "arcface_learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "max_epochs": 15,
        "validation_ratio_from_official_training": 0.15,
        "seed": 42,
    }

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "tracking_mode": "retrospective_import",
            "training_completed_before_mlflow_tracking": "true",
            "historical_run_completeness": "partial",
            "dataset_version_status": "post_training_snapshot_not_versioned_before_v1",
            "data_source": "IMSFD",
            "model_candidate": "imsfd_partial_ft_v1",
            "privacy_note": "No face photos, NIM mapping, private masterdata, or per-person manifests logged",
        })

        log_history(out / "training_history.csv")
        log_final_metrics(out / "verification_metrics_test.json")

        # Log only aggregate, non-identifying audit summary fields.
        audit_path = out / "01_audit" / "audit_summary.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            allowed = {"valid_images", "corrupt_images", "unique_identities",
                       "training_images", "testing_images", "expected_identities"}
            sanitized = {k: audit[k] for k in allowed if k in audit}
            safe_path = out / "_mlflow_safe_dataset_summary.json"
            safe_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(safe_path), artifact_path="dataset_summary")
            safe_path.unlink(missing_ok=True)

        log_file(out / "training_history.csv", "metrics")
        log_file(out / "verification_metrics_test.json", "metrics")
        log_file(out / "validation_threshold_search.csv", "metrics")
        log_dir(out / "03_checkpoints", "checkpoints")
        log_dir(out / "04_plots", "plots")
        log_dir(out / "05_model_export", "model_export")

        if args.notebook:
            log_file(Path(args.notebook).expanduser().resolve(), "source")

        print("\nImported run:", run.info.run_id)
        print("Experiment:", args.experiment)
        print("Tracking URI:", tracking_uri)
        print("Review the run in MLflow UI.")

if __name__ == "__main__":
    main()
