from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
from facenet_pytorch import MTCNN


class FaceInputError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class DetectedFace:
    tensor: torch.Tensor  # normalized float tensor [3,160,160]
    confidence: float
    bbox: list[float]
    landmarks: list[list[float]] | None
    image_width: int
    image_height: int


class MTCNNDetector:
    def __init__(self, device: torch.device, image_size=160, margin=20, min_face_size=20,
                 thresholds=(0.6, 0.7, 0.7)):
        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            keep_all=True,
            post_process=False,
            device=device,
        ).eval()

    @staticmethod
    def decode_image(data: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(data))
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.width < 64 or image.height < 64:
                raise FaceInputError("image_too_small", "Gambar terlalu kecil untuk diproses.")
            return image
        except FaceInputError:
            raise
        except Exception as exc:
            raise FaceInputError("invalid_image", "File tidak dapat dibaca sebagai gambar.") from exc

    @torch.inference_mode()
    def detect_one(self, image: Image.Image) -> DetectedFace:
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            raise FaceInputError("no_face", "Wajah tidak terdeteksi. Atur posisi wajah dan coba lagi.", {"face_count": 0})
        if len(boxes) != 1:
            raise FaceInputError("multiple_faces", "Pastikan hanya satu wajah terlihat di kamera.", {"face_count": int(len(boxes))})

        box = np.asarray(boxes[0], dtype=np.float32)
        prob = float(np.asarray(probs).reshape(-1)[0])
        landmark_list = None
        if points is not None and len(points) > 0:
            landmark_list = np.asarray(points[0], dtype=float).tolist()

        # Reuse the already detected bounding box; this avoids selecting a different face.
        crop = self.mtcnn.extract(image, np.asarray([box]), save_path=None)
        if crop is None:
            raise FaceInputError("crop_failed", "Wajah terdeteksi tetapi crop gagal dibuat.")
        if crop.ndim == 4:
            crop = crop[0]
        # MTCNN post_process=False returns 0..255 pixel tensor.
        crop = (crop.float() - 127.5) / 128.0
        return DetectedFace(
            tensor=crop.unsqueeze(0).to(self.device),
            confidence=prob,
            bbox=box.tolist(),
            landmarks=landmark_list,
            image_width=image.width,
            image_height=image.height,
        )
