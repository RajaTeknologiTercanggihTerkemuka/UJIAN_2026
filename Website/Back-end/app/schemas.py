from pydantic import BaseModel, Field
from typing import Optional


class EnrollResponse(BaseModel):
    status: str
    nim_masked: str
    model_version: str
    preprocessing_version: str
    detection_confidence: float
    embedding_dimension: int


class ParticipantRegisterResponse(BaseModel):
    request_id: str
    decision_status: str  # REGISTERED_NEW | RE_REGISTERED | REJECTED_DUPLICATE_FACE | REJECTED_NIM_FACE_MISMATCH | REJECTED_SPOOF | ACCOUNT_FLAGGED | REVIEW_REQUIRED
    nim_masked: str
    message: str


class ParticipantVerifyResponse(BaseModel):
    request_id: str
    decision_status: str  # MATCH | MISMATCH | REJECTED_SPOOF | ACCOUNT_FLAGGED | REVIEW_REQUIRED
    nim_masked: str
    message: str


class AlertReviewRequest(BaseModel):
    review_status: str = Field(..., pattern="^(pending|confirmed_joki|false_positive|resolved)$")
    reviewer_notes: Optional[str] = None
    flag_matched_user: bool = False


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
