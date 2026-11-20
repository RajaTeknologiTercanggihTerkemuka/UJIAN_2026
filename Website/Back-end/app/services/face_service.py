from __future__ import annotations
import os, time, uuid
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import User, FaceTemplate, VerificationEvent
from faceid.engine import FaceEngine, cosine_similarity
from faceid.antispoof.base import AntiSpoofResult


def mask_nim(nim: str) -> str:
    if len(nim) <= 4:
        return '*' * len(nim)
    return nim[:2] + ('*' * max(2, len(nim)-4)) + nim[-2:]


def get_current_template(db: Session, user: User, model_version: str, preprocessing_version: str):
    return (db.query(FaceTemplate)
            .filter(FaceTemplate.user_id == user.id,
                    FaceTemplate.model_version == model_version,
                    FaceTemplate.preprocessing_version == preprocessing_version)
            .order_by(FaceTemplate.created_at.desc()).first())


def save_event(db: Session, **kwargs):
    event = VerificationEvent(**kwargs)
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def verify_one_to_one(db: Session, engine: FaceEngine, antispoof_provider,
                       nim: str, image_bytes: bytes, request_id: str):
    started = time.perf_counter()
    cfg = engine.config
    user = db.query(User).filter(User.nim == nim, User.status == 'active').first()
    if not user:
        # Do not leak whether a NIM exists; log only masked NIM.
        raise HTTPException(status_code=404, detail='Data peserta tidak ditemukan atau tidak aktif.')
    template = get_current_template(db, user, cfg.model_version, cfg.preprocessing_version)
    if not template:
        raise HTTPException(status_code=409, detail='Template wajah belum tersedia untuk versi model/preprocessing aktif.')
    try:
        result = engine.embed_image_bytes(image_bytes)
    except Exception as exc:
        reason = getattr(exc, 'code', 'face_processing_error')
        elapsed = (time.perf_counter() - started) * 1000
        save_event(db, user_id=user.id, nim_masked=mask_nim(nim), mode='1:1', face_detected=False,
                   face_count=getattr(exc, 'details', {}).get('face_count', 0), decision_status='REVIEW_REQUIRED',
                   failure_reason=reason, total_latency_ms=elapsed,
                   model_version=cfg.model_version, preprocessing_version=cfg.preprocessing_version,
                   threshold_version=cfg.threshold_version, request_id=request_id)
        raise

    # PAD receives the original RGB frame and the MTCNN face box. Silent-Face applies
    # its own multi-scale expanded crop (e.g. 2.7x) from the original frame.
    try:
        anti_result = antispoof_provider.predict(result['original_rgb_for_pad'], result['bbox_xywh'])
    except Exception:
        # PAD errors must not become an accidental pass. Record generic reason without
        # exposing stack traces or frame data to the participant.
        anti_result = AntiSpoofResult(status='unknown', score=None, provider='provider_error', reason='pad_inference_error')

    if cfg.threshold is None:
        decision = 'REVIEW_REQUIRED'
        reason = 'threshold_not_calibrated'
        similarity = None
        comparison_ms = 0.0
    else:
        t_compare = time.perf_counter()
        similarity = cosine_similarity(result['embedding'], template.embedding)
        comparison_ms = (time.perf_counter() - t_compare) * 1000.0
        reason = None
        decision = 'MATCH' if similarity >= cfg.threshold else 'MISMATCH'

    require_pad = os.getenv('REQUIRE_ANTISPOOF', 'true').lower() in {'1','true','yes'}
    if require_pad and anti_result.status != 'live':
        decision = 'REVIEW_REQUIRED'
        reason = reason or ('anti_spoof_' + anti_result.status)
        if anti_result.reason == 'pad_inference_error':
            reason = 'pad_inference_error'
    elif anti_result.status == 'spoof_suspected':
        decision = 'REVIEW_REQUIRED'
        reason = 'presentation_attack_suspected'

    total_ms = (time.perf_counter() - started) * 1000.0
    save_event(db, user_id=user.id, nim_masked=mask_nim(nim), mode='1:1', face_detected=True, face_count=1,
               detection_confidence=result['detection_confidence'], face_box_area_ratio=result['face_box_area_ratio'], similarity_score=similarity,
               threshold_used=cfg.threshold, decision_status=decision, failure_reason=reason,
               quality_warnings=result['quality']['warnings'], quality_metrics={k: v for k, v in result['quality'].items() if k != 'warnings'}, antispoof_status=anti_result.status, antispoof_score=anti_result.score,
               detection_latency_ms=result['detection_latency_ms'], embedding_latency_ms=result['embedding_latency_ms'],
               comparison_latency_ms=comparison_ms, antispoof_latency_ms=anti_result.latency_ms, total_latency_ms=total_ms,
               model_version=cfg.model_version, preprocessing_version=cfg.preprocessing_version,
               threshold_version=cfg.threshold_version, request_id=request_id)

    return {
        'decision_status': decision, 'nim_masked': mask_nim(nim),
        'detection_confidence': result['detection_confidence'], 'similarity_score': similarity,
        'threshold': cfg.threshold, 'quality_warnings': result['quality']['warnings'],
        'quality_metrics': {k: v for k, v in result['quality'].items() if k != 'warnings'},
        'face_box_area_ratio': result['face_box_area_ratio'],
        'anti_spoof_status': anti_result.status, 'anti_spoof_score': anti_result.score, 'anti_spoof_latency_ms': anti_result.latency_ms, 'detection_latency_ms': result['detection_latency_ms'],
        'embedding_latency_ms': result['embedding_latency_ms'], 'comparison_latency_ms': comparison_ms,
        'total_latency_ms': total_ms, 'model_version': cfg.model_version,
        'preprocessing_version': cfg.preprocessing_version, 'threshold_version': cfg.threshold_version,
        'message': {
            'MATCH': 'Wajah cocok dengan identitas yang diklaim.',
            'MISMATCH': 'Wajah tidak cocok dengan identitas yang diklaim.',
            'REVIEW_REQUIRED': 'Verifikasi memerlukan pemeriksaan tambahan/operator.',
        }[decision]
    }
