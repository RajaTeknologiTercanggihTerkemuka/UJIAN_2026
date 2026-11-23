from __future__ import annotations
from datetime import datetime, timezone
import os
import uuid
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, FaceTemplate
from app.security import require_admin_key
from app.services.face_service import deactivate_user_templates, mask_nim, register_participant

router = APIRouter()
MAX_UPLOAD = int(os.getenv('MAX_UPLOAD_MB', '8')) * 1024 * 1024


@router.post('/register')
async def register_self_with_antijoki(
    request: Request,
    nim: str = Form(...),
    full_name: str = Form(...),
    exam_year: int = Form(2026),
    active_liveness_passed: bool = Form(True),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Participant self-registration with MTCNN, Active+PAD anti-spoofing, and 1:N Anti-Joki search."""
    if not getattr(request.app.state, 'model_ready', False):
        raise HTTPException(503, detail='Layanan registrasi belum siap. Periksa model dan konfigurasi.')
    payload = await image.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, detail='Ukuran file melebihi batas upload.')
    request_id = str(uuid.uuid4())
    try:
        detailed = register_participant(
            db=db,
            engine=request.app.state.face_engine,
            antispoof_provider=request.app.state.antispoof,
            vector_index=request.app.state.vector_index,
            nim=nim,
            full_name=full_name,
            exam_year=exam_year,
            image_bytes=payload,
            request_id=request_id,
            active_liveness_passed=active_liveness_passed,
        )
        return {
            'request_id': request_id,
            'decision_status': detailed['decision_status'],
            'nim_masked': detailed['nim_masked'],
            'message': detailed['message'],
        }
    except HTTPException:
        raise
    except Exception as exc:
        code = getattr(exc, 'code', None)
        if code:
            raise HTTPException(422, detail={'code': code, 'message': str(exc)}) from exc
        raise


@router.post('/enroll')
async def enroll_reference(
    request: Request,
    nim: str = Form(...),
    full_name: str = Form(...),
    exam_year: int = Form(2026),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Admin-only enrollment from a consented/reference photo (e.g. UTBK photo or demo seed).
    A static reference photo cannot prove live presence; this endpoint does not claim PAD.
    """
    if not getattr(request.app.state, 'model_ready', False):
        raise HTTPException(503, detail='Model belum siap. Periksa FACE_MODEL_PATH.')
    payload = await image.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, detail='Ukuran file melebihi batas upload.')
    engine = request.app.state.face_engine
    try:
        result = engine.embed_image_bytes(payload)
    except Exception as exc:
        raise HTTPException(422, detail=getattr(exc, 'code', 'face_processing_error')) from exc

    nim = nim.strip()
    full_name = full_name.strip()
    user = db.query(User).filter(User.nim == nim).first()
    if user is None:
        user = User(nim=nim, full_name=full_name, exam_year=exam_year, status='active')
        db.add(user)
        db.flush()
    else:
        user.full_name = full_name
        user.exam_year = exam_year
        user.status = 'active'
        user.updated_at = datetime.now(timezone.utc)

    cfg = engine.config
    deactivate_user_templates(db, user.id, cfg.model_version, cfg.preprocessing_version)
    template = FaceTemplate(
        user_id=user.id,
        embedding=result['embedding'].astype(float).tolist(),
        model_version=cfg.model_version,
        preprocessing_version=cfg.preprocessing_version,
        source_type='admin_reference_photo',
        is_active=True,
    )
    db.add(template)
    db.commit()
    db.refresh(template)

    vector_index = getattr(request.app.state, 'vector_index', None)
    if vector_index is not None:
        vector_index.sync_new_template(
            db, template.id, user.id, user.nim, user.full_name, user.status, result['embedding']
        )

    return {
        'status': 'ENROLLED',
        'nim_masked': mask_nim(nim),
        'exam_year': user.exam_year,
        'detection_confidence': result['detection_confidence'],
        'face_quality_warnings': result['quality']['warnings'],
        'face_quality_metrics': {k: v for k, v in result['quality'].items() if k != 'warnings'},
        'model_version': cfg.model_version,
        'preprocessing_version': cfg.preprocessing_version,
        'template_id': template.id,
        'note': 'Enrollment referensi oleh operator; ini bukan bukti liveness.',
    }
