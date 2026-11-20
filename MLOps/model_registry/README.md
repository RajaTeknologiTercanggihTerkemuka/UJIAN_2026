# MLflow Model Registry

A historical V1 import that logs a raw `.pth` file is useful for provenance, but that file alone is not a fully packaged MLflow Model. To register V1 as an MLflow PyTorch model, run `register_v1_pytorch_model.py` from an environment containing the same compatible `torch`, `facenet-pytorch`, and `mlflow` stack.

The script loads the trusted V1 export, logs a PyTorch model with a tensor input signature, creates a registered model version, and assigns the `candidate` alias. Do not assign `production` until offline validation, camera verification, PAD testing, and operator review have passed.

Registry metadata tags should include model version, training dataset snapshot, inference preprocessing version, threshold version, and git commit. Do not put NIMs, photos, or user embeddings in model registry metadata.
