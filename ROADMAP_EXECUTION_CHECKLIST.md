# Roadmap execution checklist — TA v2

Gunakan urutan ini. Jangan lompat ke production deployment sebelum seluruh acceptance criteria selesai.

## 1. MLflow historical import V1

- [ ] V1 folder copied into `MLOps/imports/imsfd_partial_ft_v1/` (whitelisted artifacts only)
- [ ] PostgreSQL `mlflow_tracking` ready
- [ ] `.env` contains `MLFLOW_BACKEND_STORE_URI` and `MLFLOW_TRACKING_URI`
- [ ] MLflow server starts
- [ ] Historical V1 run appears in `TA-v2-Face-Recognition`
- [ ] Tags explicitly identify retrospective import and `unversioned_pre_dvc_snapshot`

Acceptance: UI shows V1 epochs/metrics and model/checkpoint artifacts, with no NIM/masterdata/face photographs.

## 2. DVC dataset snapshot

- [ ] Raw IMSFD data located in private storage
- [ ] UTBK reference crops located in private storage
- [ ] `dvc init`
- [ ] `dvc add` each intended dataset folder
- [ ] Private/encrypted remote configured and tested with a harmless sample
- [ ] `.dvc` pointer files and `.gitignore` committed; actual data remains outside Git

Acceptance: `dvc status` clean and `dvc push`/`dvc pull` tested with access controls. Mark the first snapshot as post-V1.

## 3. V1 model inference smoke test + threshold

- [ ] Export `.pth` copied to `Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/`
- [ ] `model_metadata.json` copied alongside
- [ ] Face API environment has compatible torch/torchvision/facenet-pytorch
- [ ] `smoke_test_model.py` yields a 512-d embedding
- [ ] MTCNN exactly-one-face behavior checked on permitted sample images
- [ ] Calibration validation manifest uses MTCNN preprocessing and known identity labels
- [ ] Threshold chosen on validation only and stored with model/preprocess version

Acceptance: enrollment and verification embeddings use the same model and inference preprocessing version; threshold not taken from the old square-pad test set.

## 4. Bulk enrollment and model lineage

- [ ] UTBK masterdata schema confirmed privately (NIM/name column names)
- [ ] Reference-photo layout confirmed (folder per NIM or a different mapping)
- [ ] Legacy `registry_faces.pt` inspected only if trusted; archived as VGGFace2 legacy
- [ ] Bulk enrollment executed with `--dry-run` first
- [ ] Each participant has one active template for selected model/preprocessing version
- [ ] Registration process records model and preprocessing versions

Acceptance: database can retrieve template for a claimed NIM; old legacy vector is not mixed with fine-tuned embedding.

## 5. PostgreSQL + FastAPI

- [ ] `face_registration` database and `face_app_user` configured separately from MLflow
- [ ] Schema/migration applied (including `antispoof_latency_ms`)
- [ ] `.env` secrets set and not committed
- [ ] Model load succeeds and `/api/health` reports ready
- [ ] `GET /api/health`, `POST /api/verification/verify`, `POST /api/registration/enroll`, `GET /api/admin/events` smoke-tested
- [ ] Participant route returns only status/message; operator route is admin-key protected
- [ ] User authentication/session, rate limiting, HTTPS and retention requirements documented before production

Acceptance: single captured image can flow through MTCNN -> embedding -> comparison -> logged event. A missing PAD threshold must not silently pass.

## 6. Dashboard and runtime metrics

- [ ] Admin dashboard loads with admin key
- [ ] Metrics include detection confidence, face count, similarity, threshold, PAD status/score, timing breakdown, model/preprocess/threshold versions
- [ ] Face-quality checks are soft warnings; only no/multiple face gates require recapture
- [ ] `/metrics` is not exposed publicly without network/auth controls

Acceptance: participant sees concise status; operator can inspect diagnostics; raw image and full NIM do not appear in generic logs.

## 7. Presentation-attack detection / anti-spoofing

- [ ] Official source cloned and license retained
- [ ] PAD weights downloaded and SHA256 recorded
- [ ] PAD inference smoke test works on camera-captured samples
- [ ] Validation capture set contains bona fide and relevant attack types
- [ ] Test captures are separated by subject/session/device from calibration set
- [ ] PAD threshold chosen on validation (e.g., min ACER with documented security trade-off)
- [ ] Test reports APCER, BPCER, ACER and confusion matrix at frozen threshold
- [ ] Provider status `unknown`/`not_configured` leads to `REVIEW_REQUIRED` when PAD required
- [ ] Human review and non-biometric fallback path documented

Acceptance: PAD is only considered provisionally enabled after camera-specific evaluation. A pretrained public PAD model is not proof of robustness for UTBK conditions.

## What is intentionally deferred to the next MLOps phase

- Automated drift detection and retraining triggers
- Prometheus/Grafana deployment and alerting
- CI/CD, staged promotion/rollback, model approval gates
- Scheduled data-quality jobs and access-controlled dataset refresh
- Production-grade model registry governance and secrets management
