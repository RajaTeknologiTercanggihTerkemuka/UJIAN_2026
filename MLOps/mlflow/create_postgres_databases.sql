-- Run in pgAdmin Query Tool as PostgreSQL administrator.
-- Replace the placeholder passwords. Run CREATE DATABASE outside a transaction.
CREATE ROLE mlflow_user WITH LOGIN PASSWORD '222555';
CREATE DATABASE mlflow_tracking OWNER mlflow_user;

CREATE ROLE face_app_user WITH LOGIN PASSWORD '222555';
CREATE DATABASE face_registration OWNER face_app_user;
