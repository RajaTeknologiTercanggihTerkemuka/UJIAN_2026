"""Adapter untuk model MiniFASNetV2 dari proyek Silent-Face-Anti-Spoofing.

Sumber upstream: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
Model input 80x80; kelas 1 pada implementasi upstream berarti bona fide/real.
Gunakan hasil hanya setelah threshold PAD dikalibrasi dengan data kamera/attack milik sistem.
"""
from __future__ import annotations
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import AntiSpoofResult


class SilentFaceMiniFASNetProvider:
    def __init__(self, weights_dir: Path, repo_root: Path, device: torch.device, live_threshold: float | None):
        self.weights_dir = Path(weights_dir)
        self.repo_root = Path(repo_root)
        self.device = device
        self.live_threshold = live_threshold
        model_file = self.repo_root / "src" / "model_lib" / "MiniFASNet.py"
        if not model_file.exists():
            raise FileNotFoundError(
                f"MiniFASNet.py tidak ditemukan di {model_file}. Clone repo Silent-Face-Anti-Spoofing ke PAD_REPO_ROOT."
            )
        spec = importlib.util.spec_from_file_location("ta_v2_silentface_minifasnet", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Tidak dapat memuat arsitektur PAD dari {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.model_factory = {
            "MiniFASNetV1": module.MiniFASNetV1,
            "MiniFASNetV2": module.MiniFASNetV2,
            "MiniFASNetV1SE": module.MiniFASNetV1SE,
            "MiniFASNetV2SE": module.MiniFASNetV2SE,
        }
        self.models = []
        self._load_weights()

    @staticmethod
    def _parse_weight_name(path: Path):
        # Upstream format: {scale}_{height}x{width}_{ModelType}.pth
        stem = path.stem
        match = re.match(r"^(?P<scale>[0-9]+(?:\.[0-9]+)?)_(?P<h>\d+)x(?P<w>\d+)_(?P<type>MiniFASNet(?:V1SE|V2SE|V1|V2))$", stem)
        if not match:
            raise ValueError(
                f"Nama weight tidak dikenali: {path.name}. Harapkan mis. 2.7_80x80_MiniFASNetV2.pth"
            )
        return float(match.group("scale")), int(match.group("h")), int(match.group("w")), match.group("type")

    @staticmethod
    def _safe_load_state(path: Path, device: torch.device):
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise ValueError(f"Format state dict PAD tidak didukung: {path}")
        if payload and next(iter(payload)).startswith("module."):
            payload = {k[7:]: v for k, v in payload.items()}
        return payload

    def _load_weights(self):
        if not self.weights_dir.exists():
            raise FileNotFoundError(f"PAD_WEIGHTS_DIR tidak ditemukan: {self.weights_dir}")
        paths = sorted(self.weights_dir.glob("*.pth"))
        if not paths:
            raise FileNotFoundError(f"Tidak ada file .pth di {self.weights_dir}")
        for path in paths:
            scale, height, width, model_type = self._parse_weight_name(path)
            if (height, width) != (80, 80):
                raise ValueError(f"Implementasi adapter ini mengharapkan model PAD 80x80; ditemukan {path.name}")
            if model_type not in self.model_factory:
                raise ValueError(f"Model PAD tidak didukung: {model_type}")
            kernel = ((height + 15) // 16, (width + 15) // 16)
            model = self.model_factory[model_type](conv6_kernel=kernel).to(self.device)
            state = self._safe_load_state(path, self.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append({
                "path": path,
                "scale": scale,
                "height": height,
                "width": width,
                "type": model_type,
                "model": model,
            })
        if not self.models:
            raise RuntimeError("Tidak ada model PAD yang berhasil dimuat")

    @staticmethod
    def _expanded_bbox(width: int, height: int, bbox_xywh: list[float], scale: float):
        x, y, box_w, box_h = [float(v) for v in bbox_xywh]
        if box_w <= 1 or box_h <= 1:
            raise ValueError("Bounding box wajah tidak valid")
        scale = min((height - 1) / box_h, min((width - 1) / box_w, scale))
        new_w, new_h = box_w * scale, box_h * scale
        center_x, center_y = x + box_w / 2.0, y + box_h / 2.0
        left, top = center_x - new_w / 2.0, center_y - new_h / 2.0
        right, bottom = center_x + new_w / 2.0, center_y + new_h / 2.0
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > width - 1:
            left -= right - width + 1
            right = width - 1
        if bottom > height - 1:
            top -= bottom - height + 1
            bottom = height - 1
        left = max(0, int(left)); top = max(0, int(top))
        right = min(width - 1, int(right)); bottom = min(height - 1, int(bottom))
        if right <= left or bottom <= top:
            raise ValueError("Crop PAD kosong setelah bounding box diperluas")
        return left, top, right, bottom

    @torch.inference_mode()
    def predict(self, original_rgb: np.ndarray, bbox_xywh: list[float]) -> AntiSpoofResult:
        started = time.perf_counter()
        if original_rgb is None or original_rgb.ndim != 3 or original_rgb.shape[2] != 3:
            return AntiSpoofResult("unknown", None, "silentface_minifasnet", "invalid_frame")
        # The upstream Silent-Face implementation was trained/evaluated with OpenCV BGR patches.
        frame_bgr = cv2.cvtColor(np.asarray(original_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        probabilities = []
        for item in self.models:
            left, top, right, bottom = self._expanded_bbox(w, h, bbox_xywh, item["scale"])
            patch = frame_bgr[top:bottom + 1, left:right + 1]
            patch = cv2.resize(patch, (item["width"], item["height"]), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(patch).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0).to(self.device)
            logits = item["model"](tensor)
            probs = F.softmax(logits, dim=1)[0]
            if probs.numel() != 3:
                raise RuntimeError(f"PAD model {item['path'].name} menghasilkan {probs.numel()} kelas, expected 3")
            probabilities.append(probs.detach())
        mean_probs = torch.stack(probabilities, dim=0).mean(dim=0)
        live_score = float(mean_probs[1].item())  # Upstream test.py: class index 1 = real face.
        elapsed = (time.perf_counter() - started) * 1000.0
        if self.live_threshold is None:
            status, reason = "unknown", "pad_threshold_not_calibrated"
        elif live_score >= self.live_threshold:
            status, reason = "live", None
        else:
            status, reason = "spoof_suspected", "live_probability_below_calibrated_threshold"
        return AntiSpoofResult(
            status=status,
            score=live_score,
            provider="silentface_minifasnet",
            reason=reason,
            latency_ms=elapsed,
        )


def create_provider():
    from .model_loader_device import choose_device_for_pad
    weights_dir = Path(os.environ.get("PAD_WEIGHTS_DIR", "Face Recognition/artifacts/antispoof/weights"))
    repo_root = Path(os.environ.get("PAD_REPO_ROOT", "Face Recognition/third_party/Silent-Face-Anti-Spoofing"))
    project_root = Path(__file__).resolve().parents[4]
    if not weights_dir.is_absolute():
        weights_dir = (project_root / weights_dir).resolve()
    if not repo_root.is_absolute():
        repo_root = (project_root / repo_root).resolve()
    threshold_raw = os.environ.get("PAD_LIVE_THRESHOLD", "").strip()
    threshold = float(threshold_raw) if threshold_raw else None
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("PAD_LIVE_THRESHOLD harus antara 0 dan 1")
    device = choose_device_for_pad(os.environ.get("FACE_DEVICE", "auto"))
    return SilentFaceMiniFASNetProvider(weights_dir, repo_root, device, threshold)
