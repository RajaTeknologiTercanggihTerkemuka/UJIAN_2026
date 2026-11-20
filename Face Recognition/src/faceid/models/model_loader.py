from __future__ import annotations
from pathlib import Path
import torch
from facenet_pytorch import InceptionResnetV1


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FACE_DEVICE=cuda tetapi CUDA tidak tersedia pada environment ini")
    return device


def load_finetuned_backbone(model_path: Path, device: torch.device) -> InceptionResnetV1:
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model export tidak ditemukan: {model_path}")
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older torch versions
        payload = torch.load(model_path, map_location="cpu")

    state = payload.get("model_state") if isinstance(payload, dict) and "model_state" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("Format checkpoint tidak dikenali: expected state_dict or {'model_state': ...}")

    model = InceptionResnetV1(pretrained=None, classify=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
