from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class FaceQuality:
    brightness_mean: float
    brightness_std: float
    blur_score: float | None
    warnings: list[str]

    def to_dict(self):
        return asdict(self)


def assess_face_quality(face_tensor: torch.Tensor) -> FaceQuality:
    """Soft quality flags only. These warnings do not reject a face by themselves."""
    arr = face_tensor.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    # Input tensor is fixed-standardized; revert approximately to pixel scale.
    rgb = np.clip((arr.transpose(1, 2, 0) * 128.0) + 127.5, 0, 255).astype(np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    blur_score = None
    warnings = []
    if cv2 is not None:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 18.0:
            warnings.append("possible_blur")
    if brightness_mean < 45:
        warnings.append("low_brightness")
    elif brightness_mean > 215:
        warnings.append("high_brightness")
    if brightness_std < 18:
        warnings.append("low_contrast")
    return FaceQuality(brightness_mean, brightness_std, blur_score, warnings)
