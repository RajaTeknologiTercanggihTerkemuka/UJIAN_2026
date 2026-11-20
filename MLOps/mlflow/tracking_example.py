"""Template for a future Colab/local training loop with live MLflow tracking.

Call log_epoch_metrics() once per epoch from the actual training notebook/script.
"""
import mlflow


def start_training_run(tracking_uri: str, experiment: str, params: dict, tags: dict | None = None):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name="imsfd_partial_ft_v2")


def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float,
                      train_acc: float | None = None, val_acc: float | None = None):
    metrics = {"train_loss": float(train_loss), "val_loss": float(val_loss)}
    if train_acc is not None: metrics["train_identity_top1"] = float(train_acc)
    if val_acc is not None: metrics["val_identity_top1"] = float(val_acc)
    mlflow.log_metrics(metrics, step=int(epoch))


def log_training_lineage(dataset_version: str, preprocessing_version: str,
                         git_commit: str, model_version: str):
    mlflow.set_tags({
        "dataset_version": dataset_version,
        "preprocessing_version": preprocessing_version,
        "git_commit": git_commit,
        "model_version": model_version,
    })
