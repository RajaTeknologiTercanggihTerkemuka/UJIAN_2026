from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import VerificationEvent
from app.security import require_admin_key

router = APIRouter()

@router.get('/events')
def recent_events(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db), _: bool = Depends(require_admin_key)):
    rows = db.query(VerificationEvent).order_by(VerificationEvent.created_at.desc()).limit(limit).all()
    return [{
        'id': x.id, 'nim_masked': x.nim_masked, 'mode': x.mode,
        'face_detected': x.face_detected, 'face_count': x.face_count,
        'detection_confidence': x.detection_confidence, 'face_box_area_ratio': x.face_box_area_ratio,
        'similarity_score': x.similarity_score, 'threshold_used': x.threshold_used,
        'decision_status': x.decision_status, 'failure_reason': x.failure_reason,
        'quality_warnings': x.quality_warnings, 'quality_metrics': x.quality_metrics, 'antispoof_status': x.antispoof_status, 'antispoof_score': x.antispoof_score,
        'detection_latency_ms': x.detection_latency_ms, 'embedding_latency_ms': x.embedding_latency_ms,
        'comparison_latency_ms': x.comparison_latency_ms, 'antispoof_latency_ms': x.antispoof_latency_ms, 'total_latency_ms': x.total_latency_ms,
        'model_version': x.model_version, 'preprocessing_version': x.preprocessing_version,
        'threshold_version': x.threshold_version, 'created_at': x.created_at.isoformat() if x.created_at else None,
    } for x in rows]
