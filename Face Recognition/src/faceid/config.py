from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | None, default: str | None = None) -> Path | None:
    raw = value or default
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class FaceConfig:
    model_path: Path | None
    model_version: str
    preprocessing_version: str
    threshold_version: str
    threshold: float | None
    dedup_threshold: float | None
    device: str
    image_size: int
    mtcnn_margin: int
    min_face_size: int
    mtcnn_thresholds: tuple[float, float, float]

    @classmethod
    def from_env(cls) -> "FaceConfig":
        threshold_raw = os.getenv("FACE_THRESHOLD", "").strip()
        threshold = float(threshold_raw) if threshold_raw else None
        dedup_raw = os.getenv("FACE_DEDUP_THRESHOLD", "").strip()
        dedup_threshold = float(dedup_raw) if dedup_raw else threshold
        threshold_items = tuple(float(x.strip()) for x in os.getenv("MTCNN_THRESHOLDS", "0.6,0.7,0.7").split(","))
        if len(threshold_items) != 3:
            raise ValueError("MTCNN_THRESHOLDS harus berisi tiga angka, contoh 0.6,0.7,0.7")
        if threshold is not None and not (-1.0 <= threshold <= 1.0):
            raise ValueError("FACE_THRESHOLD cosine harus berada pada rentang -1 sampai 1")
        if dedup_threshold is not None and not (-1.0 <= dedup_threshold <= 1.0):
            raise ValueError("FACE_DEDUP_THRESHOLD cosine harus berada pada rentang -1 sampai 1")
        return cls(
            model_path=resolve_path(os.getenv("FACE_MODEL_PATH")),
            model_version=os.getenv("FACE_MODEL_VERSION", "unversioned"),
            preprocessing_version=os.getenv("FACE_PREPROCESSING_VERSION", "mtcnn160_margin20_fixedstd_v1"),
            threshold_version=os.getenv("FACE_THRESHOLD_VERSION", "not_calibrated"),
            threshold=threshold,
            dedup_threshold=dedup_threshold,
            device=os.getenv("FACE_DEVICE", "auto"),
            image_size=int(os.getenv("FACE_IMAGE_SIZE", "160")),
            mtcnn_margin=int(os.getenv("MTCNN_MARGIN", "20")),
            min_face_size=int(os.getenv("MTCNN_MIN_FACE_SIZE", "20")),
            mtcnn_thresholds=threshold_items,
        )
