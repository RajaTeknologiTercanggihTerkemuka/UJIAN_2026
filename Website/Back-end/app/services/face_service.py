from __future__ import annotations
from datetime import datetime, timezone
import os
import re
import time
from sqlalchemy.orm import Session

NIM_PATTERN = re.compile(r'^\d{8,9}$')


def validate_nim_format(nim: str) -> str:
    clean = nim.strip()
    if not NIM_PATTERN.match(clean):
        raise HTTPException(status_code=422, detail='NIM harus terdiri dari 8 hingga 9 digit angka (contoh: 04221001).')
    return clean
from fastapi import HTTPException
from app.models import User, FaceTemplate, VerificationEvent, DuplicateIdentityAlert
from app.services.vector_search import VectorSearchIndex
from faceid.engine import FaceEngine, cosine_similarity
from faceid.antispoof.base import AntiSpoofResult


def mask_nim(nim: str) -> str:
    if len(nim) <= 4:
        return '*' * len(nim)
    return nim[:2] + ('*' * max(2, len(nim) - 4)) + nim[-2:]


def get_current_template(db: Session, user: User, model_version: str, preprocessing_version: str):
    return (
        db.query(FaceTemplate)
        .filter(
            FaceTemplate.user_id == user.id,
            FaceTemplate.model_version == model_version,
            FaceTemplate.preprocessing_version == preprocessing_version,
            FaceTemplate.is_active.is_(True),
        )
        .order_by(FaceTemplate.created_at.desc())
        .first()
    )


def deactivate_user_templates(db: Session, user_id: int, model_version: str, preprocessing_version: str) -> None:
    (
        db.query(FaceTemplate)
        .filter(
            FaceTemplate.user_id == user_id,
            FaceTemplate.model_version == model_version,
            FaceTemplate.preprocessing_version == preprocessing_version,
            FaceTemplate.is_active.is_(True),
        )
        .update({FaceTemplate.is_active: False}, synchronize_session=False)
    )


def save_event(db: Session, **kwargs) -> VerificationEvent:
    event = VerificationEvent(**kwargs)
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def save_alert(db: Session, **kwargs) -> DuplicateIdentityAlert:
    alert = DuplicateIdentityAlert(**kwargs)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def run_antispoof(antispoof_provider, embed_result: dict) -> AntiSpoofResult:
    """Run PAD on the original RGB frame + MTCNN box. Errors never silently pass."""
    try:
        return antispoof_provider.predict(embed_result['original_rgb_for_pad'], embed_result['bbox_xywh'])
    except Exception:
        return AntiSpoofResult(
            status='unknown',
            score=None,
            provider='provider_error',
            reason='pad_inference_error',
        )


def _require_antispoof() -> bool:
    return os.getenv('REQUIRE_ANTISPOOF', 'true').lower() in {'1', 'true', 'yes'}


def register_participant(
    db: Session,
    engine: FaceEngine,
    antispoof_provider,
    vector_index: VectorSearchIndex,
    nim: str,
    full_name: str,
    exam_year: int,
    image_bytes: bytes,
    request_id: str,
    active_liveness_passed: bool = True,
) -> dict:
    """Self-registration with MTCNN single-face gate, Active+PAD anti-spoofing, and 1:N Anti-Joki search."""
    started = time.perf_counter()
    cfg = engine.config
    nim = validate_nim_format(nim)
    full_name = full_name.strip()
    if not full_name:
        raise HTTPException(status_code=422, detail='Nama lengkap wajib diisi.')

    existing_user = db.query(User).filter(User.nim == nim).first()
    if existing_user is not None and existing_user.status != 'active':
        elapsed = (time.perf_counter() - started) * 1000.0
        save_event(
            db,
            user_id=existing_user.id,
            nim_masked=mask_nim(nim),
            mode='1:N_register',
            face_detected=False,
            face_count=0,
            decision_status='ACCOUNT_FLAGGED',
            failure_reason=f'account_status_{existing_user.status}',
            total_latency_ms=elapsed,
            search_backend=vector_index.backend,
            candidate_count=vector_index.candidate_count,
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            threshold_version=cfg.threshold_version,
            request_id=request_id,
        )
        return {
            'decision_status': 'ACCOUNT_FLAGGED',
            'nim_masked': mask_nim(nim),
            'message': 'Akun ini sedang ditandai/tidak aktif. Hubungi panitia untuk verifikasi manual.',
        }

    try:
        result = engine.embed_image_bytes(image_bytes)
    except Exception as exc:
        reason = getattr(exc, 'code', 'face_processing_error')
        elapsed = (time.perf_counter() - started) * 1000.0
        save_event(
            db,
            user_id=existing_user.id if existing_user else None,
            nim_masked=mask_nim(nim),
            mode='1:N_register',
            face_detected=False,
            face_count=getattr(exc, 'details', {}).get('face_count', 0),
            decision_status='CAPTURE_FAILED',
            failure_reason=reason,
            total_latency_ms=elapsed,
            search_backend=vector_index.backend,
            candidate_count=vector_index.candidate_count,
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            threshold_version=cfg.threshold_version,
            request_id=request_id,
        )
        raise

    anti_result = run_antispoof(antispoof_provider, result)
    search_res = vector_index.search(db, result['embedding'], top_k=5)

    best_other = next((m for m in search_res.matches if m.nim != nim), None)
    existing_template = (
        get_current_template(db, existing_user, cfg.model_version, cfg.preprocessing_version)
        if existing_user is not None
        else None
    )
    own_similarity = (
        cosine_similarity(result['embedding'], existing_template.embedding)
        if existing_template is not None
        else None
    )

    verify_threshold = cfg.threshold
    dedup_threshold = cfg.dedup_threshold
    require_pad = _require_antispoof()

    decision: str
    reason: str | None = None
    matched_user_id: int | None = None
    matched_nim_masked: str | None = None
    reported_similarity: float | None = best_other.similarity if best_other is not None else own_similarity
    threshold_applied: float | None = dedup_threshold
    alert_type: str | None = None
    enrolled_user: User | None = existing_user

    if not active_liveness_passed:
        decision = 'REJECTED_SPOOF'
        reason = 'active_liveness_failed'
        alert_type = 'spoof_on_register'
        if best_other is not None:
            matched_user_id = best_other.user_id
            matched_nim_masked = mask_nim(best_other.nim)
    elif anti_result.status == 'spoof_suspected':
        decision = 'REJECTED_SPOOF'
        reason = 'presentation_attack_suspected'
        alert_type = 'spoof_on_register'
        if best_other is not None:
            matched_user_id = best_other.user_id
            matched_nim_masked = mask_nim(best_other.nim)
    elif dedup_threshold is None or verify_threshold is None:
        decision = 'REVIEW_REQUIRED'
        reason = 'threshold_not_calibrated'
    elif best_other is not None and best_other.similarity >= dedup_threshold:
        # Anti-Joki Case 1 (1:N): Face matches another NIM already stored in database!
        decision = 'REJECTED_DUPLICATE_FACE'
        reason = 'duplicate_face_detected_different_nim'
        matched_user_id = best_other.user_id
        matched_nim_masked = mask_nim(best_other.nim)
        reported_similarity = best_other.similarity
        threshold_applied = dedup_threshold
        alert_type = 'duplicate_face_different_nim'
    elif existing_template is not None and own_similarity is not None and own_similarity < verify_threshold:
        # Anti-Joki Case 2 (1:1): Different person trying to overwrite an existing NIM!
        decision = 'REJECTED_NIM_FACE_MISMATCH'
        reason = 'nim_already_registered_with_different_face'
        matched_user_id = existing_user.id
        matched_nim_masked = mask_nim(existing_user.nim)
        reported_similarity = own_similarity
        threshold_applied = verify_threshold
        alert_type = 'nim_takeover_attempt'
    elif require_pad and anti_result.status != 'live':
        decision = 'REVIEW_REQUIRED'
        reason = 'pad_inference_error' if anti_result.reason == 'pad_inference_error' else ('anti_spoof_' + anti_result.status)
    elif existing_template is not None and own_similarity is not None and own_similarity >= verify_threshold:
        # Legitimate re-registration by the same person
        deactivate_user_templates(db, existing_user.id, cfg.model_version, cfg.preprocessing_version)
        existing_user.full_name = full_name
        existing_user.exam_year = exam_year
        existing_user.updated_at = datetime.now(timezone.utc)
        template = FaceTemplate(
            user_id=existing_user.id,
            embedding=result['embedding'].astype(float).tolist(),
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            source_type='self_registration_re_enroll',
            is_active=True,
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        vector_index.sync_new_template(
            db, template.id, existing_user.id, existing_user.nim, existing_user.full_name, existing_user.status, result['embedding']
        )
        decision = 'RE_REGISTERED'
        matched_user_id = existing_user.id
        matched_nim_masked = mask_nim(existing_user.nim)
        reported_similarity = own_similarity
        threshold_applied = verify_threshold
    else:
        # Legitimate new registration
        if enrolled_user is None:
            enrolled_user = User(nim=nim, full_name=full_name, exam_year=exam_year, status='active')
            db.add(enrolled_user)
            db.flush()
        else:
            enrolled_user.full_name = full_name
            enrolled_user.exam_year = exam_year
            enrolled_user.updated_at = datetime.now(timezone.utc)

        deactivate_user_templates(db, enrolled_user.id, cfg.model_version, cfg.preprocessing_version)
        template = FaceTemplate(
            user_id=enrolled_user.id,
            embedding=result['embedding'].astype(float).tolist(),
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            source_type='self_registration_live',
            is_active=True,
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        vector_index.sync_new_template(
            db, template.id, enrolled_user.id, enrolled_user.nim, enrolled_user.full_name, enrolled_user.status, result['embedding']
        )
        decision = 'REGISTERED_NEW'

    total_ms = (time.perf_counter() - started) * 1000.0
    event = save_event(
        db,
        user_id=enrolled_user.id if enrolled_user is not None else None,
        nim_masked=mask_nim(nim),
        mode='1:N_register',
        face_detected=True,
        face_count=1,
        detection_confidence=result['detection_confidence'],
        face_box_area_ratio=result['face_box_area_ratio'],
        similarity_score=reported_similarity,
        threshold_used=threshold_applied,
        decision_status=decision,
        failure_reason=reason,
        quality_warnings=result['quality']['warnings'],
        quality_metrics={k: v for k, v in result['quality'].items() if k != 'warnings'},
        antispoof_status=anti_result.status,
        antispoof_score=anti_result.score,
        detection_latency_ms=result['detection_latency_ms'],
        embedding_latency_ms=result['embedding_latency_ms'],
        comparison_latency_ms=search_res.search_latency_ms,
        antispoof_latency_ms=anti_result.latency_ms,
        total_latency_ms=total_ms,
        matched_user_id=matched_user_id,
        matched_nim_masked=matched_nim_masked,
        search_backend=search_res.search_backend,
        candidate_count=search_res.candidate_count,
        model_version=cfg.model_version,
        preprocessing_version=cfg.preprocessing_version,
        threshold_version=cfg.threshold_version,
        request_id=request_id,
    )

    if alert_type is not None:
        save_alert(
            db,
            event_id=event.id,
            claimed_nim=nim,
            claimed_nim_masked=mask_nim(nim),
            claimed_name=full_name,
            matched_user_id=matched_user_id,
            matched_nim_masked=matched_nim_masked,
            similarity_score=reported_similarity,
            threshold_used=threshold_applied,
            alert_type=alert_type,
            review_status='pending',
            model_version=cfg.model_version,
        )

    messages = {
        'REGISTERED_NEW': 'Registrasi berhasil! Wajah Anda terdaftar sebagai peserta baru.',
        'RE_REGISTERED': 'Daftar ulang berhasil: wajah cocok dengan data Anda sebelumnya dan template wajah diperbarui.',
        'REJECTED_DUPLICATE_FACE': 'Registrasi ditolak: wajah terdeteksi sudah terdaftar pada identitas lain (indikasi duplikasi/joki). Hubungi panitia untuk verifikasi manual.',
        'REJECTED_NIM_FACE_MISMATCH': 'Registrasi ditolak: NIM ini sudah terdaftar atas wajah yang berbeda. Hubungi panitia jika ini kesalahan data.',
        'REJECTED_SPOOF': 'Registrasi ditolak: terdeteksi indikasi foto/layar/topeng (presentation attack). Gunakan wajah asli di depan kamera.',
        'REVIEW_REQUIRED': 'Registrasi memerlukan pemeriksaan tambahan/operator sebelum diaktifkan.',
    }

    return {
        'decision_status': decision,
        'nim_masked': mask_nim(nim),
        'matched_nim_masked': matched_nim_masked,
        'similarity_score': reported_similarity,
        'threshold': threshold_applied,
        'search_backend': search_res.search_backend,
        'candidate_count': search_res.candidate_count,
        'comparison_latency_ms': search_res.search_latency_ms,
        'anti_spoof_status': anti_result.status,
        'anti_spoof_score': anti_result.score,
        'total_latency_ms': total_ms,
        'message': messages[decision],
    }


def verify_one_to_one(
    db: Session,
    engine: FaceEngine,
    antispoof_provider,
    nim: str,
    image_bytes: bytes,
    request_id: str,
    full_name: str | None = None,
    active_liveness_passed: bool = True,
) -> dict:
    started = time.perf_counter()
    cfg = engine.config
    nim = validate_nim_format(nim)
    user = db.query(User).filter(User.nim == nim).first()
    if not user:
        # Do not leak whether a NIM exists; log only masked NIM.
        raise HTTPException(status_code=404, detail='Data peserta tidak ditemukan atau tidak aktif.')
    if full_name is not None and full_name.strip():
        if user.full_name.strip().lower() != full_name.strip().lower():
            raise HTTPException(status_code=422, detail='Nama tidak sesuai dengan NIM yang terdaftar.')
    if user.status != 'active':
        elapsed = (time.perf_counter() - started) * 1000.0
        save_event(
            db,
            user_id=user.id,
            nim_masked=mask_nim(nim),
            mode='1:1',
            face_detected=False,
            face_count=0,
            decision_status='ACCOUNT_FLAGGED',
            failure_reason=f'account_status_{user.status}',
            total_latency_ms=elapsed,
            search_backend='1:1_direct',
            candidate_count=1,
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            threshold_version=cfg.threshold_version,
            request_id=request_id,
        )
        return {
            'decision_status': 'ACCOUNT_FLAGGED',
            'nim_masked': mask_nim(nim),
            'message': 'Akun peserta sedang ditandai/tidak aktif. Hubungi panitia untuk verifikasi manual.',
        }

    template = get_current_template(db, user, cfg.model_version, cfg.preprocessing_version)
    if not template:
        raise HTTPException(status_code=409, detail='Template wajah belum tersedia untuk versi model/preprocessing aktif.')
    try:
        result = engine.embed_image_bytes(image_bytes)
    except Exception as exc:
        reason = getattr(exc, 'code', 'face_processing_error')
        elapsed = (time.perf_counter() - started) * 1000.0
        save_event(
            db,
            user_id=user.id,
            nim_masked=mask_nim(nim),
            mode='1:1',
            face_detected=False,
            face_count=getattr(exc, 'details', {}).get('face_count', 0),
            decision_status='REVIEW_REQUIRED',
            failure_reason=reason,
            total_latency_ms=elapsed,
            search_backend='1:1_direct',
            candidate_count=1,
            model_version=cfg.model_version,
            preprocessing_version=cfg.preprocessing_version,
            threshold_version=cfg.threshold_version,
            request_id=request_id,
        )
        raise

    # PAD receives the original RGB frame and the MTCNN face box. Silent-Face applies
    # its own multi-scale expanded crop (e.g. 2.7x) from the original frame.
    anti_result = run_antispoof(antispoof_provider, result)

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

    require_pad = _require_antispoof()
    alert_type: str | None = None
    if not active_liveness_passed:
        decision = 'REJECTED_SPOOF'
        reason = 'active_liveness_failed'
        alert_type = 'spoof_on_verify'
    elif anti_result.status == 'spoof_suspected':
        decision = 'REJECTED_SPOOF'
        reason = 'presentation_attack_suspected'
        alert_type = 'spoof_on_verify'
    elif require_pad and anti_result.status != 'live':
        decision = 'REVIEW_REQUIRED'
        reason = reason or ('anti_spoof_' + anti_result.status)
        if anti_result.reason == 'pad_inference_error':
            reason = 'pad_inference_error'

    total_ms = (time.perf_counter() - started) * 1000.0
    event = save_event(
        db,
        user_id=user.id,
        nim_masked=mask_nim(nim),
        mode='1:1',
        face_detected=True,
        face_count=1,
        detection_confidence=result['detection_confidence'],
        face_box_area_ratio=result['face_box_area_ratio'],
        similarity_score=similarity,
        threshold_used=cfg.threshold,
        decision_status=decision,
        failure_reason=reason,
        quality_warnings=result['quality']['warnings'],
        quality_metrics={k: v for k, v in result['quality'].items() if k != 'warnings'},
        antispoof_status=anti_result.status,
        antispoof_score=anti_result.score,
        detection_latency_ms=result['detection_latency_ms'],
        embedding_latency_ms=result['embedding_latency_ms'],
        comparison_latency_ms=comparison_ms,
        antispoof_latency_ms=anti_result.latency_ms,
        total_latency_ms=total_ms,
        matched_user_id=user.id if decision == 'MATCH' else None,
        matched_nim_masked=mask_nim(nim) if decision == 'MATCH' else None,
        search_backend='1:1_direct',
        candidate_count=1,
        model_version=cfg.model_version,
        preprocessing_version=cfg.preprocessing_version,
        threshold_version=cfg.threshold_version,
        request_id=request_id,
    )

    if alert_type is not None:
        save_alert(
            db,
            event_id=event.id,
            claimed_nim=nim,
            claimed_nim_masked=mask_nim(nim),
            claimed_name=user.full_name,
            matched_user_id=user.id,
            matched_nim_masked=mask_nim(nim),
            similarity_score=similarity,
            threshold_used=cfg.threshold,
            alert_type=alert_type,
            review_status='pending',
            model_version=cfg.model_version,
        )

    return {
        'decision_status': decision,
        'nim_masked': mask_nim(nim),
        'detection_confidence': result['detection_confidence'],
        'similarity_score': similarity,
        'threshold': cfg.threshold,
        'quality_warnings': result['quality']['warnings'],
        'quality_metrics': {k: v for k, v in result['quality'].items() if k != 'warnings'},
        'face_box_area_ratio': result['face_box_area_ratio'],
        'anti_spoof_status': anti_result.status,
        'anti_spoof_score': anti_result.score,
        'anti_spoof_latency_ms': anti_result.latency_ms,
        'detection_latency_ms': result['detection_latency_ms'],
        'embedding_latency_ms': result['embedding_latency_ms'],
        'comparison_latency_ms': comparison_ms,
        'total_latency_ms': total_ms,
        'model_version': cfg.model_version,
        'preprocessing_version': cfg.preprocessing_version,
        'threshold_version': cfg.threshold_version,
        'message': {
            'MATCH': 'Wajah cocok dengan identitas yang diklaim.',
            'MISMATCH': 'Wajah tidak cocok dengan identitas yang diklaim.',
            'REJECTED_SPOOF': 'Verifikasi ditolak: terdeteksi indikasi foto/layar/topeng (presentation attack).',
            'REVIEW_REQUIRED': 'Verifikasi memerlukan pemeriksaan tambahan/operator.',
        }[decision],
    }
