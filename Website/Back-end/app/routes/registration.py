from __future__ import annotations
import os, time, uuid
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, FaceTemplate
from app.security import require_admin_key
from app.services.face_service import mask_nim

router = APIRouter()
MAX_UPLOAD = int(os.getenv('MAX_UPLOAD_MB', '8')) * 1024 * 1024

@router.post('/enroll')
async def enroll_reference(
    request: Request,
    nim: str = Form(...),
    full_name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: bool = Depends(require_admin_key),
):
    """Admin-only enrollment from a consented/reference photo (e.g. UTBK photo).
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

    user = db.query(User).filter(User.nim == nim).first()
    if user is None:
        user = User(nim=nim, full_name=full_name, status='active')
        db.add(user); db.flush()
    else:
        user.full_name = full_name
        user.status = 'active'

    cfg = engine.config
    template = FaceTemplate(
        user_id=user.id,
        embedding=result['embedding'].astype(float).tolist(),
        model_version=cfg.model_version,
        preprocessing_version=cfg.preprocessing_version,
        source_type='admin_reference_photo',
    )
    db.add(template); db.commit()
    return {
        'status': 'ENROLLED', 'nim_masked': mask_nim(nim),
        'detection_confidence': result['detection_confidence'],
        'face_quality_warnings': result['quality']['warnings'],
        'face_quality_metrics': {k: v for k, v in result['quality'].items() if k != 'warnings'},
        'model_version': cfg.model_version,
        'preprocessing_version': cfg.preprocessing_version,
        'template_id': template.id,
        'note': 'Enrollment referensi oleh operator; ini bukan bukti liveness.'
    }
