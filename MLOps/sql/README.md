# PostgreSQL database bootstrap

1. Connect to maintenance DB `postgres` as a PostgreSQL admin.
2. Edit and run `01_create_roles.sql` once. Replace the two passwords. If role exists, skip its CREATE ROLE line.
3. Run `02_create_mlflow_database.sql` as a separate single statement. `CREATE DATABASE` must be executed outside a transaction block.
4. Run `03_create_app_database.sql` as a separate single statement.
5. Configure `.env`: `MLFLOW_BACKEND_STORE_URI` points to `mlflow_tracking`; `DATABASE_URL` points to `face_registration`. Use URL-encoded passwords if they contain URI-reserved characters.

MLflow backend metadata and app data are separate. MLflow artifact files stay in a separate private artifact directory or approved object store.
