from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class AntiSpoofResult:
    status: str  # live | spoof_suspected | unknown | not_configured
    score: float | None
    provider: str
    reason: str | None = None
    latency_ms: float | None = None


class AntiSpoofProvider(Protocol):
    def predict(self, original_rgb: np.ndarray, bbox_xywh: list[float]) -> AntiSpoofResult: ...


class UnconfiguredAntiSpoofProvider:
    """Safe default: it never claims that a face is live."""
    def predict(self, original_rgb: np.ndarray, bbox_xywh: list[float]) -> AntiSpoofResult:
        return AntiSpoofResult(
            status="not_configured", score=None, provider="unconfigured",
            reason="No validated presentation-attack-detection model is configured.",
        )


def load_provider_from_env():
    """Load MODULE:CALLABLE factory. Never silently fall back to a fake/live mock."""
    import os, importlib
    target = os.getenv("ANTI_SPOOF_PROVIDER", "").strip()
    if not target:
        return UnconfiguredAntiSpoofProvider()
    if ":" not in target:
        raise ValueError("ANTI_SPOOF_PROVIDER harus berbentuk MODULE:CALLABLE")
    module_name, callable_name = target.split(":", 1)
    factory = getattr(importlib.import_module(module_name), callable_name)
    provider = factory()
    if not hasattr(provider, "predict"):
        raise TypeError("Anti-spoof provider harus memiliki method predict(original_rgb, bbox_xywh)")
    return provider
