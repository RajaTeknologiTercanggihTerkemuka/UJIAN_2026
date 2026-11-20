from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd
import mlflow
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SAFE_ARTIFACT_NAMES = {
    "training_history.csv", "verification_metrics_test.json",
    "validation_threshold_search.csv", "loss_curve.png",
    "arcface_top1_curve.png", "classification_accuracy_curve.png",
    "confusion_matrix_verification_test.png", "roc_curve_verification_test.png",
    "inceptionresnetv1_imsfd_partial_finetuned.pth",
    "best_checkpoint.pth", "last_checkpoint.pth", "model_metadata.json",
}

def first_existing(root: Path, filename: str):
    matches = list(root.rglob(filename))
    return matches[0] if matches else None

def safe_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def main():
    parser = argparse.ArgumentParser(description="Import completed V1 run into MLflow as retrospective tracking.")
    parser.add_argument("--source-dir", required=True, help="Folder copied from IMSFD_ArcFace_Output_v2")
    parser.add_argument("--experiment", default="TA-v2-Face-Recognition")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    history_path = first_existing(source, "training_history.csv")
    metrics_path = first_existing(source, "verification_metrics_test.json")
    threshold_path = first_existing(source, "validation_threshold_search.csv")

    with mlflow.start_run(run_name="imsfd_partial_ft_v1_retrospective") as run:
        mlflow.set_tags({
            "model_version": "imsfd_partial_ft_v1",
            "tracking_type": "retrospective_import",
            "dataset": "IMSFD",
            "backbone": "InceptionResnetV1",
            "initial_weights": "VGGFace2",
            "dataset_version": "unversioned_pre_dvc_snapshot",
            "dvc_status_at_training_time": "not_used_before_v1_training",
            "training_objective": "ArcFace",
            "preprocessing_version": "squarepad_resize160_fixedstd_v1",
            "note": "Imported after training; MLflow did not record the original live training process.",
        })
        mlflow.log_params({
            "architecture": "InceptionResnetV1",
            "embedding_dim": 512,
            "input_size": 160,
            "arcface_margin": 0.50,
            "arcface_scale": 64.0,
            "unfrozen_modules": "repeat_3,block8,last_linear,last_bn",
        })

        if history_path:
            df = pd.read_csv(history_path)
            mlflow.log_artifact(str(history_path), artifact_path="historical_metrics")
            for _, row in df.iterrows():
                step = int(row.get("epoch", 0))
                aliases = {
                    "train_loss": ["train_loss"],
                    "val_loss": ["val_loss", "validation_loss"],
                    "train_top1": ["train_arcface_top1", "train_acc", "train_accuracy"],
                    "val_top1": ["val_arcface_top1", "val_acc", "val_accuracy"],
                }
                for metric_name, candidates in aliases.items():
                    for col in candidates:
                        if col in df.columns:
                            value = safe_float(row[col])
                            if value is not None:
                                mlflow.log_metric(metric_name, value, step=step)
                            break
        else:
            print("WARNING: training_history.csv not found; no per-epoch metrics can be reconstructed.")

        if metrics_path:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
            for key, value in data.items():
                number = safe_float(value)
                if number is not None and isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{key}" if not key.startswith("test_") else key, number)
            threshold = data.get("threshold_from_validation", data.get("threshold_selected_on_validation"))
            if threshold is not None:
                mlflow.log_param("historical_validation_threshold", str(threshold))
        else:
            print("WARNING: verification_metrics_test.json not found.")

        if threshold_path:
            mlflow.log_artifact(str(threshold_path), artifact_path="evaluation")

        # Log only explicitly allowlisted files. Never log data manifests, identity mappings,
        # student photos, masterdata, registry_faces.pt, or private CSV files.
        logged = set()
        for filename in SAFE_ARTIFACT_NAMES:
            path = first_existing(source, filename)
            if path and path.is_file() and path.resolve() not in logged:
                mlflow.log_artifact(str(path), artifact_path="model_and_reports")
                logged.add(path.resolve())

        print("Imported historical run:", run.info.run_id)
        print("Tracking URI:", tracking_uri)
        print("Artifacts imported:", len(logged))
        print("Reminder: retrospective tracking cannot restore data/compute logs that were never recorded.")

if __name__ == "__main__":
    main()
