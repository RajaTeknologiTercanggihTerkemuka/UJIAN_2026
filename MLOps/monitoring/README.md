# Monitoring design

**MLflow** is for training experiments/model lineage. It is not a per-request live dashboard by itself.

**PostgreSQL `verification_events`** stores per-request operational facts: masked NIM, scores, threshold, decision, latencies, version tags, quality warnings, and PAD status. No raw frames or full embeddings are stored by default.

**Dashboard operator** reads a protected API endpoint for recent events. Keep score visibility restricted to authorized operator/research roles. Participants see only clear status/instructions.

Later, expose Prometheus metrics and use Grafana if needed. Monitor p50/p95 latency, request errors, no-face/multi-face rates, score distributions, PAD review rate, CPU/RAM/GPU. FAR/FRR cannot be computed from unlabelled production events; they require trusted ground-truth adjudication or a controlled labelled evaluation sample.

Do not automatically retrain from model predictions or treat a model's own `MATCH` as a correct label. Require verified labels and review.
