# MLOps — TA v2

## Komponen

- **Git**: source code, config, notebook (tanpa data pribadi/berkas bobot besar).
- **DVC**: snapshot/versioning dataset dan hasil preprocessing. V1 sudah dilatih sebelum DVC dipasang; snapshot pertama adalah pencatatan retrospektif, bukan bukti lineage saat training V1.
- **MLflow**: tracking run, parameter, per-epoch metrics jika tersedia, evaluation metrics, artifacts, model registration.
- **PostgreSQL `mlflow_tracking`**: backend metadata MLflow.
- **Artifact store**: folder lokal privat atau remote privat; jangan letakkan data biometrik atau identity mapping dalam artifacts MLflow.
- **PostgreSQL `face_registration`**: database aplikasi yang terpisah.

## Urutan awal

1. Import V1 historical run dari output Colab (tidak retraining).
2. Snapshot raw IMSFD dan UTBK reference photos menggunakan DVC ke remote privat/encrypted.
3. Register export model V1 as candidate.
4. Smoke-test backbone + MTCNN + preprocessing and recalibrate threshold on validation data.
5. Recompute UTBK templates using the exact selected model/preprocessing version; old `registry_faces.pt` remains legacy.
6. Continue with API, operator dashboard, and PAD.

Jangan mengklaim MLflow/DVC melakukan tracking real-time pada training V1. MLflow hanya mengimpor artefak yang tersedia; DVC mulai digunakan setelah V1.
