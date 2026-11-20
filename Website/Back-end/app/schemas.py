from pydantic import BaseModel
from typing import Optional

class EnrollResponse(BaseModel):
    status: str
    nim_masked: str
    model_version: str
    preprocessing_version: str
    detection_confidence: float
    embedding_dimension: int

class VerifyResponse(BaseModel):
    decision_status: str
    nim_masked: str
    detection_confidence: Optional[float] = None
    similarity_score: Optional[float] = None
    threshold: Optional[float] = None
    quality_warnings: list[str] = []
    anti_spoof_status: str
    detection_latency_ms: Optional[float] = None
    embedding_latency_ms: Optional[float] = None
    anti_spoof_latency_ms: Optional[float] = None
    comparison_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    model_version: str
    preprocessing_version: str
    threshold_version: str
    message: str
