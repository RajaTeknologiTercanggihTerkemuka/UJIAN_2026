# TA v2 — Roadmap Implementation Pack

Paket ini adalah overlay untuk proyek TA v2 yang sudah Anda susun. Ia menambahkan contoh kode, konfigurasi, SQL, MLOps, API, operator dashboard, dan integrasi PAD. Paket **tidak** menyertakan dataset wajah, masterdata, `registry_faces.pt`, model InceptionResnetV1, atau model PAD weights.

## Status penting

- V1 hasil Colab dapat diimpor ke MLflow secara retrospektif. Tidak perlu fine-tuning ulang hanya untuk menggunakan MLflow.
- DVC mulai sekarang. Snapshot pertamanya adalah post-training snapshot, bukan data lineage yang sudah terekam saat V1.
- Model export V1 perlu disalin dari Drive ke `Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/`.
- `registry_faces.pt` berisi embedding dari model sebelum fine-tuning: arsipkan dan jangan campur dengan embedding V1. Foto UTBK tetap dapat dipakai sebagai reference enrollment setelah embedding dihitung ulang dengan model/preprocessing aktif.
- Foto UTBK yang sudah di-crop dapat dipakai sebagai reference enrollment setelah di-embed ulang dengan model/preprocessing yang dipilih.
- MTCNN mendeteksi/crop wajah; PAD (MiniFASNet) terpisah. PAD belum dianggap tervalidasi hanya karena weights berhasil dimuat.
- Ambang similarity dan PAD harus dikalibrasi di validation data dan dibekukan sebelum evaluasi test. Foto UTBK statis bukan data bona fide live untuk mengkalibrasi PAD; gunakan sesi capture kamera sukarela plus presentasi serangan (print/replay) yang terpisah.
- API participant hanya menampilkan status; diagnostic score ada pada operator dashboard dengan akses admin.

## Urutan jalan yang disarankan

### 0. Kondisi folder VS Code saat ini

Pada screenshot proyek, `.venv-mlops` sudah berada di root dan starter kit lama masih nested di `MLOps/TA v2 MLOps Starter Kit/MLOps`. Gabungkan folder `MLOps/` dari paket overlay langsung ke root proyek `TA v2/MLOps/`; jangan menaruhnya lagi dalam subfolder starter kit. Jangan hapus `.venv-mlops`, `.venv-face-api` (jika sudah dibuat), atau backup `TA v2 Original Copy`.

## 1. Gabungkan paket ke folder proyek kerja

Ekstrak ZIP ke folder sementara. Salin isi overlay ke root `TA v2`, bukan ke `TA v2 Original Copy`. Pada screenshot sebelumnya, Starter Kit berada bertingkat di `MLOps/TA v2 MLOps Starter Kit/MLOps`; ambil isi folder `MLOps` dari overlay dan letakkan langsung di `TA v2/MLOps/`. Jangan overwrite `.venv-mlops`.

### 2. MLflow V1 historical import

- Gunakan `.venv-mlops` yang sudah dibuat.
- Install `Requirement/requirements-mlops.txt`.
- Siapkan `mlflow_tracking` PostgreSQL dan `.env`.
- Jalankan `python MLOps/mlflow/start_mlflow.py`.
- Salin V1 output tanpa raw dataset, identity mapping, masterdata, foto, atau registry_faces.pt ke `MLOps/imports/imsfd_partial_ft_v1/`.
- Jalankan `python MLOps/mlflow/import_v1_run.py --source-dir MLOps/imports/imsfd_partial_ft_v1`.

### 3. DVC snapshot

Baca `MLOps/data_versioning/README.md`. Gunakan remote privat/encrypted. Jangan menganggap DVC dapat mengenkripsi file otomatis.

### 4. Model V1 + threshold smoke test

- Salin export `.pth` ke `Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/`.
- Gunakan environment Face API yang kompatibel dengan Torch/torchvision/facenet-pytorch.
- Jalankan `Face Recognition/scripts/smoke_test_model.py` pada gambar uji yang diizinkan.
- Threshold V1 hanya berlaku untuk pipeline V1; jika preprocessing berubah, recalibrate pada validation pairs.

### 5. PostgreSQL, reference enrollment, FastAPI

- Buat database `face_registration` terpisah dari `mlflow_tracking`.
- `.env` wajib memiliki `ADMIN_API_KEY`, `DATABASE_URL`, `FACE_MODEL_PATH`, model/preprocessing version, dan threshold terkalibrasi.
- Import masterdata hanya pada lingkungan privat; bulk enrollment harus dry-run dulu dan memakai foto per NIM folder sesuai script. Sesuaikan pemetaan jika folder Anda tidak per NIM.
- Jalankan API: `uvicorn app.main:app --app-dir "Website/Back-end" --reload`.
- `/` peserta, `/admin` operator, `/metrics` Prometheus prototype.
- Belum production ready: peserta belum terhubung SSO/login, rate limit/auth hardening/HTTPS belum lengkap.

### 6. MTCNN + dashboard operator

MTCNN menerima frame asli, memeriksa tepat satu wajah, membuat crop 160x160 untuk model. Face quality warnings bersifat lunak; no-face/multiple-face meminta capture ulang. Operator melihat confidence, similarity, threshold, PAD status, latency, model dan preprocessing versions.

### 7. Anti-spoofing/PAD

- Candidate: official `minivision-ai/Silent-Face-Anti-Spoofing`, repo Apache-2.0; official README menjelaskan print/display/silicone/3D presentation attacks, model MiniFASNet input 80x80, and that camera/device conditions influence performance.
- Run `Face Recognition/scripts/download_silentface_weights.ps1` from PowerShell (after reviewing upstream source/license). It clones the repo and downloads the official MiniFASNetV2 2.7x 80x80 weight to a local weights directory. The helper `predict_pad_manifest.py` uses the project root two levels above its `scripts/` folder.
- Configure `ANTI_SPOOF_PROVIDER=faceid.antispoof.silentface_provider:create_provider`, `PAD_REPO_ROOT`, `PAD_WEIGHTS_DIR`; leave `PAD_LIVE_THRESHOLD` empty until calibrated.
- Build PAD validation/test sets with bona fide camera-captured sessions and separate presentation attacks (print photo, phone/tablet replay; add masks if in scope). Split by subjects/session/device; don't tune on test.
- Use `predict_pad_manifest.py` then `evaluate_pad_threshold.py` to choose threshold on validation and report test APCER/BPCER/ACER. Model's upstream test uses class index 1 as real; this provider follows that convention.
- When threshold is blank, endpoint returns `REVIEW_REQUIRED` with `REQUIRE_ANTISPOOF=true`. Do not change this to false just to make demo pass.
- This open-source pretrained PAD is a candidate baseline; it is not guaranteed robust for your webcam/phone or local attack types. Validate it on your own capture protocol. Blink/head-turn is an additional challenge, not a substitute for PAD.

## Local commands (PowerShell)

```powershell
# From TA v2 root. .venv-mlops exists in your current VS Code screenshot.
.\.venv-mlops\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r Requirement\requirements-mlops.txt
python MLOps\mlflow\start_mlflow.py
```

Use a second terminal for historical import. Do not run all environments' requirements in one venv. Install torch/torchvision separately for the face API environment using a compatible version selected for the machine.

## Important environment separation

- `.venv-mlops`: MLflow, DVC, PostgreSQL client dependencies, and historical tracking scripts only.
- `.venv-face-api`: FastAPI + facenet-pytorch/MTCNN + PAD. Install PyTorch/Torchvision separately using the official selector for the machine's CPU/CUDA; test the `facenet-pytorch` compatibility before installing the whole set.
- Do not install `requirements-mlops.txt`, `requirements-face-api.txt`, and `requirements-antispoof.txt` into one environment by default.
- The operator dashboard/API in this pack is a research prototype. It uses a local admin API key, has no complete account/session authentication, rate limiting, production TLS setup, or hardened key management. Keep it localhost-only while testing.
- The included front-end camera captures a still image on button click. It is not continuous real-time frame streaming; WebSocket/continuous sampling can be a later enhancement.
- The PAD provider adapter is wired to the official Silent-Face source and MiniFASNetV2 weights, but the actual upstream model must still be downloaded and evaluated with your target camera and attack protocol. Do not present `live` as an established security guarantee before this validation.

## Critical limitations

The code is a research prototype. A successful HTTP response, loaded model, or PAD score is not proof of operational biometric reliability. Before real examination use, perform independent validation for demographic/device/camera conditions, presentation attacks, failure handling, human review, participant authentication, security, consent and retention policies, and authorized legal review. Automated face decision should not be the sole basis for a consequential exam decision.
