# Privacy and retention rules for logs

- Do not write full NIM, name, raw image, embedding vector, or camera stream into application logs, MLflow params, or Prometheus labels.
- Use masked identifiers for operator event tables; protect database access and encrypt storage/backups.
- Keep images in memory for request processing unless the research protocol explicitly requires retention and participants consent.
- Record model/preprocessing/threshold versions and operational metrics without retaining image payloads.
- Establish retention/deletion periods and a manual/alternative identity verification path.
- Treat face embeddings as biometric data, not as anonymous harmless telemetry.
