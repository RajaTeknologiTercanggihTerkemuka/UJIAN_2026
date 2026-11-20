from __future__ import annotations
import torch

def choose_device_for_pad(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FACE_DEVICE=cuda tetapi CUDA tidak tersedia")
    return device
