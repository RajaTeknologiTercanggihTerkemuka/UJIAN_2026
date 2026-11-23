from __future__ import annotations
import os, sys, logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / '.env', override=False)
sys.path.insert(0, str(PROJECT_ROOT / 'Face Recognition' / 'src'))

from faceid.config import FaceConfig
from faceid.engine import FaceEngine
from faceid.antispoof.base import load_provider_from_env
from app.database import Base, engine, SessionLocal
from app.services.vector_search import VectorSearchIndex
from app.routes import registration, verification, admin

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('ta_v2.api')
HTTP_REQUESTS = Counter('ta_v2_http_requests_total', 'HTTP requests', ['method', 'route', 'status'])
HTTP_LATENCY = Histogram('ta_v2_http_request_seconds', 'HTTP request duration', ['method', 'route'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        app.state.face_config = FaceConfig.from_env()
        app.state.face_engine = FaceEngine(app.state.face_config)
        app.state.antispoof = load_provider_from_env()
        vector_index = VectorSearchIndex(
            model_version=app.state.face_config.model_version,
            preprocessing_version=app.state.face_config.preprocessing_version,
        )
        with SessionLocal() as db:
            index_info = vector_index.load_from_db(db)
        app.state.vector_index = vector_index
        app.state.model_ready = True
        logger.info(
            'Face engine loaded: %s | vector_backend=%s | indexed_vectors=%d',
            app.state.face_config.model_version,
            index_info['backend'],
            index_info['indexed_vectors'],
        )
    except Exception as exc:
        app.state.model_ready = False
        app.state.startup_error = str(exc)
        logger.exception('Face engine not ready. Check .env and model export.')
    yield


app = FastAPI(title='TA v2 Face Registration & Verification API', version='0.2.0', lifespan=lifespan)
origins = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5173,http://localhost:5173').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['*'])


@app.middleware('http')
async def request_metrics(request, call_next):
    import time
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    route_obj = request.scope.get('route')
    route = getattr(route_obj, 'path', 'unmatched')
    HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method, route).observe(duration)
    return response


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(registration.router, prefix='/api/registration', tags=['registration'])
app.include_router(verification.router, prefix='/api/verification', tags=['verification'])
app.include_router(admin.router, prefix='/api/admin', tags=['admin'])


@app.get('/api/health')
def health():
    ready = bool(getattr(app.state, 'model_ready', False))
    vector_index = getattr(app.state, 'vector_index', None)
    return {
        'status': 'ready' if ready else 'not_ready',
        'model_ready': ready,
        'startup_error': None if ready else getattr(app.state, 'startup_error', 'Model not loaded'),
        'model_version': getattr(getattr(app.state, 'face_config', None), 'model_version', None),
        'preprocessing_version': getattr(getattr(app.state, 'face_config', None), 'preprocessing_version', None),
        'vector_backend': getattr(vector_index, 'backend', None),
        'indexed_vectors': getattr(vector_index, 'candidate_count', 0),
    }


STATIC_DIR = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def index():
    from fastapi.responses import FileResponse
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/admin')
def admin_page():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url='/?tab=admin', status_code=302)
