import os, hmac
from fastapi import Header, HTTPException, status

def require_admin_key(x_admin_key: str | None = Header(default=None)):
    expected = os.getenv('ADMIN_API_KEY', '')
    if not expected or expected.startswith('CHANGE_'):
        raise HTTPException(status_code=503, detail='ADMIN_API_KEY belum dikonfigurasi')
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Admin key tidak valid')
    return True
