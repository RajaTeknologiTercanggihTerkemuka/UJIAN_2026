from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.request
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import DuplicateIdentityAlert, FaceTemplate, User, VerificationEvent
from app.schemas import AlertReviewRequest
from app.security import require_admin_key

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _read_json_safe(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return None


def _parse_dvc_pointer(path: Path) -> dict | None:
    if not path.exists():
        return None
    info: dict[str, str | int | None] = {'file': path.name, 'path': None, 'nfiles': None, 'size_bytes': None, 'md5': None}
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s.startswith('- md5:') or s.startswith('md5:'):
            info['md5'] = s.split(':', 1)[1].strip()
        elif s.startswith('size:'):
            try:
                info['size_bytes'] = int(s.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith('nfiles:'):
            try:
                info['nfiles'] = int(s.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith('path:'):
            info['path'] = s.split(':', 1)[1].strip()
    return info


def _check_mlflow_server(tracking_uri: str) -> bool:
    if not tracking_uri or not tracking_uri.startswith('http'):
        return False
    try:
        req = urllib.request.Request(tracking_uri.rstrip('/') + '/health', method='GET')
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            return resp.status == 200
    except Exception:
        try:
            req = urllib.request.Request(tracking_uri.rstrip('/') + '/', method='GET')
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                return resp.status < 500
        except Exception:
            return False


@router.get('/events')
def recent_events(
    limit: int = Query(100, ge=1, le=500),
    mode: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    q = db.query(VerificationEvent)
    if mode:
        q = q.filter(VerificationEvent.mode == mode)
    rows = q.order_by(VerificationEvent.created_at.desc()).limit(limit).all()
    return [
        {
            'id': x.id,
            'nim_masked': x.nim_masked,
            'mode': x.mode,
            'face_detected': x.face_detected,
            'face_count': x.face_count,
            'detection_confidence': x.detection_confidence,
            'face_box_area_ratio': x.face_box_area_ratio,
            'similarity_score': x.similarity_score,
            'threshold_used': x.threshold_used,
            'decision_status': x.decision_status,
            'failure_reason': x.failure_reason,
            'quality_warnings': x.quality_warnings,
            'quality_metrics': x.quality_metrics,
            'antispoof_status': x.antispoof_status,
            'antispoof_score': x.antispoof_score,
            'detection_latency_ms': x.detection_latency_ms,
            'embedding_latency_ms': x.embedding_latency_ms,
            'comparison_latency_ms': x.comparison_latency_ms,
            'antispoof_latency_ms': x.antispoof_latency_ms,
            'total_latency_ms': x.total_latency_ms,
            'matched_nim_masked': x.matched_nim_masked,
            'search_backend': x.search_backend,
            'candidate_count': x.candidate_count,
            'model_version': x.model_version,
            'preprocessing_version': x.preprocessing_version,
            'threshold_version': x.threshold_version,
            'created_at': x.created_at.isoformat() if x.created_at else None,
        }
        for x in rows
    ]


@router.get('/alerts')
def list_duplicate_alerts(
    review_status: str | None = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """List Anti-Joki duplicate identity, NIM takeover, and presentation-attack alerts."""
    q = db.query(DuplicateIdentityAlert)
    if review_status:
        q = q.filter(DuplicateIdentityAlert.review_status == review_status)
    rows = q.order_by(DuplicateIdentityAlert.created_at.desc()).limit(limit).all()
    return [
        {
            'id': a.id,
            'event_id': a.event_id,
            'claimed_nim_masked': a.claimed_nim_masked,
            'claimed_name': a.claimed_name,
            'matched_user_id': a.matched_user_id,
            'matched_nim_masked': a.matched_nim_masked,
            'similarity_score': a.similarity_score,
            'threshold_used': a.threshold_used,
            'alert_type': a.alert_type,
            'review_status': a.review_status,
            'reviewer_notes': a.reviewer_notes,
            'model_version': a.model_version,
            'created_at': a.created_at.isoformat() if a.created_at else None,
            'reviewed_at': a.reviewed_at.isoformat() if a.reviewed_at else None,
        }
        for a in rows
    ]


@router.post('/alerts/{alert_id}/review')
def review_duplicate_alert(
    alert_id: int,
    payload: AlertReviewRequest,
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    alert = db.query(DuplicateIdentityAlert).filter(DuplicateIdentityAlert.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=404, detail='Alert tidak ditemukan.')
    alert.review_status = payload.review_status
    alert.reviewer_notes = payload.reviewer_notes
    alert.reviewed_at = datetime.now(timezone.utc)

    flagged_user_id = None
    if payload.flag_matched_user and alert.matched_user_id is not None:
        matched_user = db.query(User).filter(User.id == alert.matched_user_id).first()
        if matched_user is not None:
            matched_user.status = 'flagged_joki'
            matched_user.updated_at = datetime.now(timezone.utc)
            flagged_user_id = matched_user.id

    db.commit()
    return {
        'id': alert.id,
        'review_status': alert.review_status,
        'flagged_user_id': flagged_user_id,
        'reviewed_at': alert.reviewed_at.isoformat() if alert.reviewed_at else None,
    }


@router.get('/stats')
def runtime_stats(
    request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Operational + MLOps passive drift telemetry summary for operator dashboard."""
    cfg = getattr(request.app.state, 'face_config', None)
    vector_index = getattr(request.app.state, 'vector_index', None)

    total_users = db.query(func.count(User.id)).scalar() or 0
    active_users = db.query(func.count(User.id)).filter(User.status == 'active').scalar() or 0
    flagged_users = db.query(func.count(User.id)).filter(User.status == 'flagged_joki').scalar() or 0
    active_templates = db.query(func.count(FaceTemplate.id)).filter(FaceTemplate.is_active.is_(True)).scalar() or 0

    status_counts_rows = (
        db.query(VerificationEvent.decision_status, func.count(VerificationEvent.id))
        .group_by(VerificationEvent.decision_status)
        .all()
    )
    decision_counts = {str(status): int(cnt) for status, cnt in status_counts_rows}

    pending_alerts = (
        db.query(func.count(DuplicateIdentityAlert.id))
        .filter(DuplicateIdentityAlert.review_status == 'pending')
        .scalar()
        or 0
    )

    avg_row = (
        db.query(
            func.avg(VerificationEvent.similarity_score),
            func.avg(VerificationEvent.detection_confidence),
            func.avg(VerificationEvent.comparison_latency_ms),
            func.avg(VerificationEvent.total_latency_ms),
            func.avg(VerificationEvent.antispoof_score),
        )
        .filter(VerificationEvent.face_detected.is_(True))
        .first()
    )

    return {
        'users': {
            'total': int(total_users),
            'active': int(active_users),
            'flagged_joki': int(flagged_users),
        },
        'templates': {
            'active_total': int(active_templates),
        },
        'vector_index': {
            'backend': getattr(vector_index, 'backend', None),
            'indexed_vectors': getattr(vector_index, 'candidate_count', 0),
        },
        'model': {
            'model_version': getattr(cfg, 'model_version', None),
            'preprocessing_version': getattr(cfg, 'preprocessing_version', None),
            'threshold_version': getattr(cfg, 'threshold_version', None),
            'verify_threshold': getattr(cfg, 'threshold', None),
            'dedup_threshold': getattr(cfg, 'dedup_threshold', None),
            'pad_live_threshold': os.getenv('PAD_LIVE_THRESHOLD', '').strip() or None,
            'require_antispoof': os.getenv('REQUIRE_ANTISPOOF', 'true').lower() in {'1', 'true', 'yes'},
        },
        'events': {
            'total': sum(decision_counts.values()),
            'by_decision': decision_counts,
            'pending_alerts': int(pending_alerts),
        },
        'telemetry_averages': {
            'avg_similarity_score': float(avg_row[0]) if avg_row and avg_row[0] is not None else None,
            'avg_detection_confidence': float(avg_row[1]) if avg_row and avg_row[1] is not None else None,
            'avg_comparison_latency_ms': float(avg_row[2]) if avg_row and avg_row[2] is not None else None,
            'avg_total_latency_ms': float(avg_row[3]) if avg_row and avg_row[3] is not None else None,
            'avg_antispoof_score': float(avg_row[4]) if avg_row and avg_row[4] is not None else None,
        },
    }


@router.get('/mlops')
def mlops_overview(
    request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Comprehensive MLOps panel payload: MLflow status, model/PAD/DVC artifacts, training history, and live drift/retraining trigger."""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000').strip()
    mlflow_online = _check_mlflow_server(tracking_uri)

    model_meta = _read_json_safe(
        PROJECT_ROOT / 'Face Recognition' / 'artifacts' / 'exported_models' / 'imsfd_partial_ft_v1' / 'model_metadata.json'
    )
    face_threshold = _read_json_safe(
        PROJECT_ROOT / 'Face Recognition' / 'artifacts' / 'thresholds' / 'threshold_v1.json'
    )
    pad_threshold = _read_json_safe(
        PROJECT_ROOT / 'Face Recognition' / 'artifacts' / 'thresholds' / 'pad_threshold_v1.json'
    )
    test_metrics = _read_json_safe(
        PROJECT_ROOT / 'MLOps' / 'imports' / 'imsfd_partial_ft_v1' / 'verification_metrics_test.json'
    )
    audit_summary = _read_json_safe(
        PROJECT_ROOT / 'MLOps' / 'imports' / 'imsfd_partial_ft_v1' / '01_audit' / 'audit_summary.json'
    )

    training_history: list[dict] = []
    hist_path = PROJECT_ROOT / 'MLOps' / 'imports' / 'imsfd_partial_ft_v1' / 'training_history.csv'
    if hist_path.exists():
        try:
            with hist_path.open('r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    training_history.append(
                        {
                            'epoch': int(float(row['epoch'])),
                            'train_loss': round(float(row['train_loss']), 4),
                            'val_loss': round(float(row['val_loss']), 4),
                            'train_top1': round(float(row['train_arcface_top1']), 4),
                            'val_top1': round(float(row['val_arcface_top1']), 4),
                        }
                    )
        except Exception:
            training_history = []

    dvc_datasets = [
        x
        for x in [
            _parse_dvc_pointer(PROJECT_ROOT / 'Dataset' / 'raw' / 'IMSFD_original.dvc'),
        ]
        if x is not None
    ]

    # Live Passive Drift & Retraining Advisory Trigger computed from recent calibrated verification_events
    # Exclude pre-calibration events (e.g. anti_spoof_unknown before PAD_LIVE_THRESHOLD was set) so drift reflects real runtime health.
    recent_raw = (
        db.query(VerificationEvent)
        .order_by(VerificationEvent.created_at.desc())
        .limit(150)
        .all()
    )
    recent = [
        e
        for e in recent_raw
        if (e.failure_reason or '') not in {'anti_spoof_unknown', 'anti_spoof_not_configured', 'threshold_not_calibrated'}
    ][:100]
    sample_n = len(recent)
    match_sims = [
        e.similarity_score
        for e in recent
        if e.decision_status in {'MATCH', 'RE_REGISTERED'} and e.similarity_score is not None
    ]
    pad_scores = [e.antispoof_score for e in recent if e.antispoof_score is not None]
    latencies = [e.total_latency_ms for e in recent if e.total_latency_ms is not None]
    failure_events = [
        e for e in recent if e.decision_status in {'MISMATCH', 'CAPTURE_FAILED'}
    ]
    spoof_events = [e for e in recent if e.decision_status == 'REJECTED_SPOOF']
    joki_events = [
        e for e in recent if e.decision_status in {'REJECTED_DUPLICATE_FACE', 'REJECTED_NIM_FACE_MISMATCH'}
    ]

    avg_match_sim = (sum(match_sims) / len(match_sims)) if match_sims else None
    avg_pad_score = (sum(pad_scores) / len(pad_scores)) if pad_scores else None
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    failure_rate = (len(failure_events) / sample_n) if sample_n else 0.0
    spoof_rate = (len(spoof_events) / sample_n) if sample_n else 0.0

    # Drift evaluation rules
    reasons: list[str] = []
    status = 'HEALTHY'
    if sample_n < 5:
        reasons.append(f'Sampel operasional masih awal ({sample_n} event). Model V1 aktif dan stabil pada baseline terkalibrasi.')
    else:
        if avg_match_sim is not None and avg_match_sim < 0.50:
            status = 'RETRAINING_RECOMMENDED'
            reasons.append(f'Rata-rata similarity MATCH turun drastis ({avg_match_sim:.3f} < 0.50) — indikasi domain/lighting drift.')
        elif avg_match_sim is not None and avg_match_sim < 0.65:
            if status != 'RETRAINING_RECOMMENDED':
                status = 'DRIFT_WARNING'
            reasons.append(f'Rata-rata similarity MATCH menurun ({avg_match_sim:.3f} < 0.65) — pantau kondisi kamera/pencahayaan.')

        if failure_rate >= 0.45:
            status = 'RETRAINING_RECOMMENDED'
            reasons.append(f'Rasio kegagalan/mismatch tinggi ({failure_rate * 100:.1f}% >= 45%).')
        elif failure_rate >= 0.25:
            if status != 'RETRAINING_RECOMMENDED':
                status = 'DRIFT_WARNING'
            reasons.append(f'Rasio kegagalan/mismatch meningkat ({failure_rate * 100:.1f}% >= 25%).')

        if avg_latency is not None and avg_latency > 2000.0:
            if status == 'HEALTHY':
                status = 'DRIFT_WARNING'
            reasons.append(f'Rata-rata latency tinggi ({avg_latency:.0f} ms > 2000 ms) — periksa beban CPU/VPS.')

        if not reasons:
            reasons.append('Distribusi skor similarity, tingkat kegagalan, dan latency berada dalam batas normal.')

    return {
        'mlflow': {
            'tracking_uri': tracking_uri,
            'public_url': os.getenv('MLFLOW_PUBLIC_URL', tracking_uri).strip(),
            'online': mlflow_online,
            'experiment_name': 'TA-v2-Face-Recognition',
            'imported_run_id': '8215ce4898564b3c8bf585b2719155a9',
        },
        'model_metadata': model_meta,
        'audit_summary': audit_summary,
        'verification_test_metrics': test_metrics,
        'face_threshold_calibration': face_threshold,
        'pad_threshold_calibration': pad_threshold,
        'training_history': training_history,
        'dvc_datasets': dvc_datasets,
        'drift_monitor': {
            'status': status,
            'window_events': sample_n,
            'match_samples': len(match_sims),
            'avg_match_similarity': round(avg_match_sim, 4) if avg_match_sim is not None else None,
            'failure_rate': round(failure_rate, 4),
            'spoof_rate': round(spoof_rate, 4),
            'joki_blocked_count': len(joki_events),
            'avg_pad_score': round(avg_pad_score, 4) if avg_pad_score is not None else None,
            'avg_total_latency_ms': round(avg_latency, 1) if avg_latency is not None else None,
            'reasons': reasons,
            'retraining_policy': (
                'Retraining otomatis dinonaktifkan demi keamanan biometrik (mencegah data poisoning) dan kepatuhan privasi (UU PDP). '
                'Jika status RETRAINING_RECOMMENDED muncul, operator melakukan kurasi data berizin lalu menjalankan pipeline fine-tuning V2 dan mencatat eksperimen baru di MLflow.'
            ),
        },
    }


@router.get('/demo-users')
def list_demo_users(
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """List ONLY seeded DEMO2026* accounts and self-registered demo participants (never exposes bulk UTBK PII)."""
    allowed_sources = {'demo_seed_reference', 'self_registration_live', 'self_registration_re_enroll'}
    rows = (
        db.query(User, FaceTemplate.source_type)
        .join(FaceTemplate, FaceTemplate.user_id == User.id)
        .filter(
            FaceTemplate.is_active.is_(True),
            (User.nim.like('DEMO2026%')) | (FaceTemplate.source_type.in_(allowed_sources)),
        )
        .order_by(User.created_at.desc())
        .limit(50)
        .all()
    )
    seen: set[int] = set()
    result = []
    for u, src in rows:
        if u.id in seen:
            continue
        seen.add(u.id)
        result.append(
            {
                'id': u.id,
                'nim': u.nim,
                'full_name': u.full_name,
                'exam_year': u.exam_year,
                'status': u.status,
                'source_type': src,
                'created_at': u.created_at.isoformat() if u.created_at else None,
            }
        )
    return result


@router.delete('/users/{nim}')
def delete_user_by_nim(
    nim: str,
    request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Delete a test/demo user by NIM and refresh the 1:N vector index (helpful after demo rehearsal)."""
    clean_nim = nim.strip()
    user = db.query(User).filter(User.nim == clean_nim).first()
    if user is None:
        raise HTTPException(status_code=404, detail=f'Peserta dengan NIM {clean_nim} tidak ditemukan.')
    uid = user.id
    db.query(DuplicateIdentityAlert).filter(
        (DuplicateIdentityAlert.claimed_nim == clean_nim) | (DuplicateIdentityAlert.matched_user_id == uid)
    ).delete(synchronize_session=False)
    db.query(FaceTemplate).filter(FaceTemplate.user_id == uid).delete(synchronize_session=False)
    db.delete(user)
    db.commit()

    vector_index = getattr(request.app.state, 'vector_index', None)
    index_info = vector_index.load_from_db(db) if vector_index is not None else {}
    return {'status': 'deleted', 'nim': clean_nim, 'vector_index': index_info}


@router.delete('/events')
def reset_events_and_alerts(
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Clear all verification/registration event history and duplicate alerts without deleting users or face templates."""
    deleted_alerts = db.query(DuplicateIdentityAlert).delete(synchronize_session=False)
    deleted_events = db.query(VerificationEvent).delete(synchronize_session=False)
    db.commit()
    return {
        'status': 'reset',
        'deleted_events': int(deleted_events),
        'deleted_alerts': int(deleted_alerts),
    }


@router.post('/index/reload')
def reload_vector_index(
    request: Request,
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    vector_index = getattr(request.app.state, 'vector_index', None)
    if vector_index is None:
        raise HTTPException(status_code=503, detail='Vector index belum diinisialisasi.')
    info = vector_index.load_from_db(db)
    return {'status': 'reloaded', **info}
