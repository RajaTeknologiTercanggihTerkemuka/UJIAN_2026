# MLflow

`start_mlflow.py` starts a local tracking server. PostgreSQL stores run/experiment metadata; model/checkpoint/plot files live in the local artifact directory `MLOps/mlflow/artifacts/`.

For current MLflow, a tracking server exposes UI + APIs and supports SQLAlchemy backend stores such as PostgreSQL. Keep it bound to localhost for development; do not expose unauthenticated tracking UI or PII artifacts to the network.


For PostgreSQL, set `MLFLOW_BACKEND_STORE_URI=postgresql+psycopg://mlflow_user:...@127.0.0.1:5432/mlflow_tracking` in the private `.env`. If blank, `start_mlflow.py` uses a local SQLite database for a first smoke test. Do not store the password in scripts or Git.
