-- Run once in pgAdmin connected to a maintenance database (e.g. postgres).
-- Replace both passwords. If roles already exist, skip their CREATE ROLE statements.
CREATE ROLE mlflow_user LOGIN PASSWORD 'REPLACE_WITH_STRONG_MLFLOW_PASSWORD';
CREATE ROLE face_app_user LOGIN PASSWORD 'REPLACE_WITH_DIFFERENT_APP_PASSWORD';
