from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from .config import FaceConfig
from .models.model_loader import choose_device, load_finetuned_backbone
from .detection.mtcnn_detector import MTCNNDetector
from .quality import assess_face_quality


class FaceEngine:
    def __init__(self, config: FaceConfig):
        self.config = config
        self.device = choose_device(config.device)
        if config.model_path is None:
            raise FileNotFoundError("FACE_MODEL_PATH belum diatur pada .env")
        self.model = load_finetuned_backbone(config.model_path, self.device)
        self.detector = MTCNNDetector(
            device=self.device,
            image_size=config.image_size,
            margin=config.mtcnn_margin,
            min_face_size=config.min_face_size,
            thresholds=config.mtcnn_thresholds,
        )

    @torch.inference_mode()
    def embed_image_bytes(self, image_bytes: bytes) -> dict:
        image = self.detector.decode_image(image_bytes)
        # Keep original RGB frame in memory only. PAD uses it with the same face box;
        # it is not returned to the client or persisted in the database.
        original_rgb = np.asarray(image, dtype=np.uint8).copy()

        t0 = time.perf_counter()
        face = self.detector.detect_one(image)
        detection_ms = (time.perf_counter() - t0) * 1000.0

        quality = assess_face_quality(face.tensor)
        t1 = time.perf_counter()
        embedding = self.model(face.tensor)
        embedding = F.normalize(embedding.float(), p=2, dim=1)[0]
        if embedding.numel() != 512:
            raise RuntimeError(f"Dimensi embedding tidak sesuai: {embedding.numel()} (expected 512)")
        embedding_ms = (time.perf_counter() - t1) * 1000.0

        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area_ratio = (width * height) / max(1.0, face.image_width * face.image_height)

        return {
            "embedding": embedding.detach().cpu().numpy().astype(np.float32),
            "original_rgb_for_pad": original_rgb,
            "bbox_xywh": [x1, y1, width, height],
            "detection_confidence": face.confidence,
            "bbox_xyxy": face.bbox,
            "face_box_area_ratio": float(area_ratio),
            "landmarks": face.landmarks,
            "quality": quality.to_dict(),
            "detection_latency_ms": detection_ms,
            "embedding_latency_ms": embedding_ms,
            "face_count": 1,
        }


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != (512,) or b.shape != (512,):
        raise ValueError("Embedding harus memiliki 512 dimensi")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("Embedding memiliki norma nol")
    return float(np.dot(a, b) / denom)
