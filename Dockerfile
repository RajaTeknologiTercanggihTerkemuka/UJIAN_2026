FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACE_DEVICE=cpu

WORKDIR /app

# Minimal OS libraries required by OpenCV headless & PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch CPU-only first (saves >2 GB compared to CUDA wheels)
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
       torch==2.4.1+cpu torchvision==0.19.1+cpu

# 2. Install facenet-pytorch without overriding CPU torch, then VPS inference deps
COPY Requirement/requirements-vps.txt /app/Requirement/requirements-vps.txt
RUN pip install --no-deps facenet-pytorch==2.6.0 \
    && pip install -r /app/Requirement/requirements-vps.txt

# 3. Copy lightweight application, inference code, SQL migrations, and tiny MLOps JSON/CSV metadata
# Note: JSON array syntax is required by Dockerfile COPY for paths containing spaces ("Face Recognition").
COPY Website/Back-end /app/Website/Back-end
COPY ["Face Recognition/src", "/app/Face Recognition/src"]
COPY ["Face Recognition/third_party/Silent-Face-Anti-Spoofing/src", "/app/Face Recognition/third_party/Silent-Face-Anti-Spoofing/src"]
COPY ["Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/model_metadata.json", "/app/Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/model_metadata.json"]
COPY ["Face Recognition/artifacts/thresholds/threshold_v1.json", "/app/Face Recognition/artifacts/thresholds/threshold_v1.json"]
COPY ["Face Recognition/artifacts/thresholds/pad_threshold_v1.json", "/app/Face Recognition/artifacts/thresholds/pad_threshold_v1.json"]
COPY MLOps/imports/imsfd_partial_ft_v1/verification_metrics_test.json /app/MLOps/imports/imsfd_partial_ft_v1/verification_metrics_test.json
COPY MLOps/imports/imsfd_partial_ft_v1/training_history.csv /app/MLOps/imports/imsfd_partial_ft_v1/training_history.csv
COPY MLOps/imports/imsfd_partial_ft_v1/01_audit/audit_summary.json /app/MLOps/imports/imsfd_partial_ft_v1/01_audit/audit_summary.json
COPY Dataset/raw/IMSFD_original.dvc /app/Dataset/raw/IMSFD_original.dvc
COPY Dataset/raw/UTBK_reference_images.dvc /app/Dataset/raw/UTBK_reference_images.dvc
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
COPY deploy/benchmark_hnsw_vps.py /app/deploy/benchmark_hnsw_vps.py

RUN sed -i 's/\r$//' /app/deploy/entrypoint.sh \
    && chmod +x /app/deploy/entrypoint.sh \
    && mkdir -p "/app/Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1" \
                "/app/Face Recognition/artifacts/antispoof/weights"

EXPOSE 8000

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
