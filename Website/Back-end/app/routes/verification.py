from __future__ import annotations
import os, uuid
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.face_service import verify_one_to_one

router = APIRouter()
MAX_UPLOAD = int(os.getenv('MAX_UPLOAD_MB', '8')) * 1024 * 1024

@router.post('/verify')
async def verify_claimed_identity(
    request: Request,
    nim: str = Form(...),
    full_name: str = Form(''),
    active_liveness_passed: bool = Form(True),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Participant endpoint: returns a simple result. Detailed scores stay admin-only in DB."""
    if not getattr(request.app.state, 'model_ready', False):
        raise HTTPException(503, detail='Layanan verifikasi belum siap.')
    payload = await image.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, detail='Ukuran file melebihi batas upload.')
    request_id = str(uuid.uuid4())
    try:
        detailed = verify_one_to_one(
            db, request.app.state.face_engine, request.app.state.antispoof,
            nim, payload, request_id, full_name=full_name, active_liveness_passed=active_liveness_passed,
        )
        # Do not reveal similarity, threshold, PAD score, model version, or detailed
        # diagnostics in the participant-facing response.
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
