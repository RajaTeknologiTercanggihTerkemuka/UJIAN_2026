from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.models import FaceTemplate, User

logger = logging.getLogger("ta_v2.vector_search")


@dataclass(frozen=True)
class VectorMatch:
    template_id: int
    user_id: int
    nim: str
    full_name: str
    user_status: str
    similarity: float


@dataclass(frozen=True)
class VectorSearchResult:
    matches: list[VectorMatch]
    search_backend: str
    candidate_count: int
    search_latency_ms: float


def _normalize_vec(vec: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape != (512,):
        raise ValueError(f"Embedding harus 512 dimensi, diterima {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Embedding memiliki norma nol")
    return (arr / norm).astype(np.float32)


def _format_pgvector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


class VectorSearchIndex:
    """High-speed 1:N face embedding search supporting dual backends:

    1. ``pgvector_hnsw``: Native PostgreSQL ``pgvector`` HNSW cosine index (<=> operator)
       when the ``vector`` extension and ``face_templates.embedding_vec`` column are present.
    2. ``numpy_matrix_cache``: Exact BLAS-accelerated in-memory unit-normalized float32 matrix
       fallback for local PostgreSQL installations without pgvector (e.g. Windows dev or SQLite tests).
       Searches 12,876+ 512-d vectors in ~1-3 ms on CPU with ~26 MB RAM.
    """

    def __init__(self, model_version: str, preprocessing_version: str):
        self.model_version = model_version
        self.preprocessing_version = preprocessing_version
        self._lock = threading.RLock()
        self.backend: str = "numpy_matrix_cache"
        self._matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._meta: list[tuple[int, int, str, str, str]] = []  # (template_id, user_id, nim, full_name, status)
        self._pgvector_count: int = 0

    @property
    def candidate_count(self) -> int:
        with self._lock:
            if self.backend == "pgvector_hnsw":
                return self._pgvector_count
            return len(self._meta)

    def _detect_pgvector(self, db: Session) -> bool:
        bind = db.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return False
        try:
            ext = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
            if not ext:
                return False
            col = db.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'face_templates' AND column_name = 'embedding_vec'"
                )
            ).scalar()
            return bool(col)
        except Exception as exc:
            logger.warning("Deteksi pgvector gagal, menggunakan numpy_matrix_cache: %s", exc)
            return False

    def load_from_db(self, db: Session) -> dict:
        t0 = time.perf_counter()
        use_pgvector = self._detect_pgvector(db)
        with self._lock:
            if use_pgvector:
                self.backend = "pgvector_hnsw"
                self._matrix = np.empty((0, 512), dtype=np.float32)
                self._meta = []
                cnt = db.execute(
                    text(
                        "SELECT COUNT(*) FROM face_templates "
                        "WHERE model_version = :mv AND preprocessing_version = :pv "
                        "AND is_active = TRUE AND embedding_vec IS NOT NULL"
                    ),
                    {"mv": self.model_version, "pv": self.preprocessing_version},
                ).scalar()
                self._pgvector_count = int(cnt or 0)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "VectorSearchIndex aktif: backend=pgvector_hnsw, vectors=%d (%.1f ms)",
                    self._pgvector_count,
                    elapsed_ms,
                )
                return {
                    "backend": self.backend,
                    "indexed_vectors": self._pgvector_count,
                    "load_ms": elapsed_ms,
                }

            self.backend = "numpy_matrix_cache"
            rows = (
                db.query(
                    FaceTemplate.id,
                    FaceTemplate.user_id,
                    User.nim,
                    User.full_name,
                    User.status,
                    FaceTemplate.embedding,
                )
                .join(User, User.id == FaceTemplate.user_id)
                .filter(
                    FaceTemplate.model_version == self.model_version,
                    FaceTemplate.preprocessing_version == self.preprocessing_version,
                    FaceTemplate.is_active.is_(True),
                )
                .order_by(FaceTemplate.created_at.desc())
                .all()
            )

            vectors: list[np.ndarray] = []
            meta: list[tuple[int, int, str, str, str]] = []
            seen_users: set[int] = set()
            for tid, uid, nim, full_name, user_status, emb in rows:
                # Keep the most recent active template per user in memory index
                if uid in seen_users:
                    continue
                try:
                    vec = _normalize_vec(emb)
                except Exception:
                    continue
                seen_users.add(uid)
                vectors.append(vec)
                meta.append((int(tid), int(uid), str(nim), str(full_name), str(user_status)))

            self._matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 512), dtype=np.float32)
            self._meta = meta
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "VectorSearchIndex aktif: backend=numpy_matrix_cache, vectors=%d (%.1f ms)",
                len(self._meta),
                elapsed_ms,
            )
            return {
                "backend": self.backend,
                "indexed_vectors": len(self._meta),
                "load_ms": elapsed_ms,
            }

    def sync_new_template(
        self,
        db: Session,
        template_id: int,
        user_id: int,
        nim: str,
        full_name: str,
        user_status: str,
        embedding: np.ndarray | list[float],
    ) -> None:
        vec = _normalize_vec(embedding)
        with self._lock:
            if self.backend == "pgvector_hnsw":
                db.execute(
                    text("UPDATE face_templates SET embedding_vec = CAST(:vec AS vector) WHERE id = :tid"),
                    {"vec": _format_pgvector_literal(vec), "tid": int(template_id)},
                )
                db.commit()
                cnt = db.execute(
                    text(
                        "SELECT COUNT(*) FROM face_templates "
                        "WHERE model_version = :mv AND preprocessing_version = :pv "
                        "AND is_active = TRUE AND embedding_vec IS NOT NULL"
                    ),
                    {"mv": self.model_version, "pv": self.preprocessing_version},
                ).scalar()
                self._pgvector_count = int(cnt or 0)
                return

            # Update in-memory numpy matrix: replace if user_id already indexed, else append.
            replace_idx = None
            for idx, item in enumerate(self._meta):
                if item[1] == user_id:
                    replace_idx = idx
                    break
            new_meta = (int(template_id), int(user_id), str(nim), str(full_name), str(user_status))
            if replace_idx is not None:
                self._matrix[replace_idx] = vec
                self._meta[replace_idx] = new_meta
            else:
                if self._matrix.shape[0] == 0:
                    self._matrix = vec.reshape(1, 512)
                else:
                    self._matrix = np.vstack([self._matrix, vec.reshape(1, 512)])
                self._meta.append(new_meta)

    def search(self, db: Session, query_embedding: np.ndarray | list[float], top_k: int = 5) -> VectorSearchResult:
        q = _normalize_vec(query_embedding)
        t0 = time.perf_counter()

        if self.backend == "pgvector_hnsw":
            raw_limit = max(top_k * 2, 10)
            rows = db.execute(
                text(
                    "SELECT ft.id, ft.user_id, u.nim, u.full_name, u.status, "
                    "       1.0 - (ft.embedding_vec <=> CAST(:vec AS vector)) AS similarity "
                    "FROM face_templates ft "
                    "JOIN users u ON u.id = ft.user_id "
                    "WHERE ft.model_version = :mv "
                    "  AND ft.preprocessing_version = :pv "
                    "  AND ft.is_active = TRUE "
                    "  AND ft.embedding_vec IS NOT NULL "
                    "ORDER BY ft.embedding_vec <=> CAST(:vec AS vector) ASC "
                    "LIMIT :lim"
                ),
                {
                    "vec": _format_pgvector_literal(q),
                    "mv": self.model_version,
                    "pv": self.preprocessing_version,
                    "lim": raw_limit,
                },
            ).fetchall()

            matches: list[VectorMatch] = []
            seen_users: set[int] = set()
            for tid, uid, nim, full_name, user_status, sim in rows:
                uid_int = int(uid)
                if uid_int in seen_users:
                    continue
                seen_users.add(uid_int)
                matches.append(
                    VectorMatch(
                        template_id=int(tid),
                        user_id=uid_int,
                        nim=str(nim),
                        full_name=str(full_name),
                        user_status=str(user_status),
                        similarity=float(sim),
                    )
                )
                if len(matches) >= top_k:
                    break
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return VectorSearchResult(
                matches=matches,
                search_backend="pgvector_hnsw",
                candidate_count=self.candidate_count,
                search_latency_ms=elapsed_ms,
            )

        # In-memory exact cosine search via BLAS matrix-vector multiply
        with self._lock:
            n = self._matrix.shape[0]
            if n == 0:
                return VectorSearchResult(
                    matches=[],
                    search_backend="numpy_matrix_cache",
                    candidate_count=0,
                    search_latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            scores = self._matrix @ q  # shape: (n,)
            k = min(n, max(top_k, 1))
            if n > k:
                part_idx = np.argpartition(scores, -k)[-k:]
                top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
            else:
                top_idx = np.argsort(scores)[::-1]

            matches = []
            for idx in top_idx:
                tid, uid, nim, full_name, user_status = self._meta[int(idx)]
                matches.append(
                    VectorMatch(
                        template_id=tid,
                        user_id=uid,
                        nim=nim,
                        full_name=full_name,
                        user_status=user_status,
                        similarity=float(scores[int(idx)]),
                    )
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return VectorSearchResult(
                matches=matches,
                search_backend="numpy_matrix_cache",
                candidate_count=n,
                search_latency_ms=elapsed_ms,
            )
