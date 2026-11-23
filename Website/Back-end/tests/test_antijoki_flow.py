from __future__ import annotations
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Face Recognition" / "src"))

from app.database import SessionLocal
from app.main import app
from app.models import DuplicateIdentityAlert, FaceTemplate, User, VerificationEvent

TEST_NIM_PREFIX = "TESTJOKI_"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TEST_PORT = 18765
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def _http_get(path: str, headers: dict[str, str] | None = None) -> tuple[int, dict | list]:
    req = urllib.request.Request(BASE_URL + path, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _http_post_json(path: str, payload: dict, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(BASE_URL + path, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _http_post_multipart(
    path: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    boundary = "----TAv2Boundary" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
            f"{v}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode("utf-8")
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})}
    req = urllib.request.Request(BASE_URL + path, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _pick_two_identities(source_dir: Path, face_engine, vector_index) -> tuple[list[Path], list[Path]]:
    found: list[list[Path]] = []
    dedup_th = face_engine.config.dedup_threshold or 0.75
    with SessionLocal() as db:
        for sub in sorted(source_dir.iterdir()):
            if not sub.is_dir():
                continue
            imgs = sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            valid_imgs: list[Path] = []
            for img in imgs[:10]:
                try:
                    emb = face_engine.embed_image_bytes(img.read_bytes())["embedding"]
                    res = vector_index.search(db, emb, top_k=1)
                    if res.matches and res.matches[0].similarity >= dedup_th:
                        continue
                    valid_imgs.append(img)
                except Exception:
                    continue
                if len(valid_imgs) >= 2:
                    break
            if len(valid_imgs) >= 2:
                found.append(valid_imgs)
            if len(found) >= 2:
                break
    if len(found) < 2:
        raise RuntimeError(f"Butuh minimal 2 folder identitas baru (>=2 gambar single-face) di {source_dir}")
    return found[0], found[1]


def _cleanup_test_records() -> None:
    with SessionLocal() as db:
        users = db.query(User).filter(User.nim.like(f"{TEST_NIM_PREFIX}%")).all()
        ids = [u.id for u in users]
        db.query(DuplicateIdentityAlert).filter(
            DuplicateIdentityAlert.claimed_nim.like(f"{TEST_NIM_PREFIX}%")
        ).delete(synchronize_session=False)
        if ids:
            db.query(DuplicateIdentityAlert).filter(
                DuplicateIdentityAlert.matched_user_id.in_(ids)
            ).delete(synchronize_session=False)
            db.query(VerificationEvent).filter(
                (VerificationEvent.user_id.in_(ids))
                | (VerificationEvent.matched_user_id.in_(ids))
            ).delete(synchronize_session=False)
            db.query(FaceTemplate).filter(FaceTemplate.user_id.in_(ids)).delete(synchronize_session=False)
            db.query(User).filter(User.id.in_(ids)).delete(synchronize_session=False)
        db.query(VerificationEvent).filter(
            VerificationEvent.nim_masked.like("TE%*")
        ).delete(synchronize_session=False)
        db.commit()


def _start_server() -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            code, data = _http_get("/api/health")
            if code == 200 and data.get("model_ready"):
                return server, thread
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server gagal siap dalam 60 detik.")


def run_e2e_test() -> None:
    source_dir = PROJECT_ROOT / "Dataset" / "raw" / "IMSFD_original" / "A1" / "Testing"
    admin_key = os.getenv("ADMIN_API_KEY", "")
    orig_require_pad = os.environ.get("REQUIRE_ANTISPOOF")

    _cleanup_test_records()
    server = None
    thread = None
    try:
        server, thread = _start_server()
        person_a_imgs, person_b_imgs = _pick_two_identities(source_dir, app.state.face_engine, app.state.vector_index)
        a_img1 = person_a_imgs[0].read_bytes()
        a_img2 = person_a_imgs[1].read_bytes()
        b_img1 = person_b_imgs[0].read_bytes()

        # 0. Health check
        code, health = _http_get("/api/health")
        assert code == 200, health
        print("[0] /api/health:", health)
        assert health["model_ready"] is True
        assert health["vector_backend"] in {"pgvector_hnsw", "numpy_matrix_cache"}
        assert health["indexed_vectors"] > 0

        # 1. Fail-safe PAD gate test: REQUIRE_ANTISPOOF=true & blank PAD_LIVE_THRESHOLD -> REVIEW_REQUIRED
        os.environ["REQUIRE_ANTISPOOF"] = "true"
        code, d1 = _http_post_multipart(
            "/api/registration/register",
            {"nim": f"{TEST_NIM_PREFIX}PAD1", "full_name": "Test FailSafe PAD", "exam_year": "2026"},
            "image", "a1.jpg", a_img1,
        )
        assert code == 200, d1
        print("[1] Fail-safe PAD gate (uncalibrated):", d1)
        assert d1["decision_status"] == "REVIEW_REQUIRED", d1

        # Controlled non-PAD flow to verify all 1:N Anti-Joki & 1:1 verification branches
        os.environ["REQUIRE_ANTISPOOF"] = "false"

        # 2. Legitimate NEW registration (Person A -> TESTJOKI_A1)
        code, d2 = _http_post_multipart(
            "/api/registration/register",
            {"nim": f"{TEST_NIM_PREFIX}A1", "full_name": "Test Peserta A", "exam_year": "2026"},
            "image", "a1.jpg", a_img1,
        )
        assert code == 200, d2
        print("[2] New registration (Person A -> A1):", d2)
        assert d2["decision_status"] == "REGISTERED_NEW", d2

        # 3. Anti-Joki Case 1: Person A (different photo!) tries registering under DIFFERENT NIM TESTJOKI_A2
        code, d3 = _http_post_multipart(
            "/api/registration/register",
            {"nim": f"{TEST_NIM_PREFIX}A2", "full_name": "Joki Identitas B", "exam_year": "2026"},
            "image", "a2.jpg", a_img2,
        )
        assert code == 200, d3
        print("[3] Anti-Joki Duplicate Face (Person A -> A2):", d3)
        assert d3["decision_status"] == "REJECTED_DUPLICATE_FACE", d3

        # 4. Anti-Joki Case 2: Person B tries overwriting Person A's existing NIM TESTJOKI_A1
        code, d4 = _http_post_multipart(
            "/api/registration/register",
            {"nim": f"{TEST_NIM_PREFIX}A1", "full_name": "Pengambilalih NIM", "exam_year": "2026"},
            "image", "b1.jpg", b_img1,
        )
        assert code == 200, d4
        print("[4] Anti-Joki NIM Takeover (Person B -> A1):", d4)
        assert d4["decision_status"] == "REJECTED_NIM_FACE_MISMATCH", d4

        # 5. Legitimate Re-Registration: Person A re-registers under OWN NIM TESTJOKI_A1 with a_img2
        code, d5 = _http_post_multipart(
            "/api/registration/register",
            {"nim": f"{TEST_NIM_PREFIX}A1", "full_name": "Test Peserta A Updated", "exam_year": "2026"},
            "image", "a2.jpg", a_img2,
        )
        assert code == 200, d5
        print("[5] Legitimate Re-Registration (Person A -> A1):", d5)
        assert d5["decision_status"] == "RE_REGISTERED", d5

        # 6. 1:1 Verification: Genuine (Person A -> A1 = MATCH) and Impostor (Person B -> A1 = MISMATCH)
        code, d6a = _http_post_multipart(
            "/api/verification/verify",
            {"nim": f"{TEST_NIM_PREFIX}A1"},
            "image", "a1.jpg", a_img1,
        )
        assert code == 200, d6a
        print("[6a] 1:1 Verify Genuine (Person A -> A1):", d6a)
        assert d6a["decision_status"] == "MATCH", d6a

        code, d6b = _http_post_multipart(
            "/api/verification/verify",
            {"nim": f"{TEST_NIM_PREFIX}A1"},
            "image", "b1.jpg", b_img1,
        )
        assert code == 200, d6b
        print("[6b] 1:1 Verify Impostor (Person B -> A1):", d6b)
        assert d6b["decision_status"] == "MISMATCH", d6b

        # 7. Admin alerts, alert review, stats, and events endpoints
        headers = {"X-Admin-Key": admin_key}
        code, alerts = _http_get("/api/admin/alerts?limit=10", headers=headers)
        assert code == 200, alerts
        alert_types = [a["alert_type"] for a in alerts[:5]]
        print("[7a] Recent Anti-Joki alerts:", alert_types)
        assert "duplicate_face_different_nim" in alert_types
        assert "nim_takeover_attempt" in alert_types

        # Review the first test alert cleanly
        review_code, review_data = _http_post_json(
            f"/api/admin/alerts/{alerts[0]['id']}/review",
            {"review_status": "confirmed_joki", "reviewer_notes": "Verified by automated E2E test", "flag_matched_user": False},
            headers=headers,
        )
        assert review_code == 200, review_data
        print("[7b] Alert review:", review_data)

        code, stats = _http_get("/api/admin/stats", headers=headers)
        assert code == 200, stats
        print("[7c] Admin stats:", json.dumps(stats, indent=2))
        assert stats["vector_index"]["indexed_vectors"] >= 1

        code, events = _http_get("/api/admin/events?limit=8", headers=headers)
        assert code == 200, events
        reg_events = [e for e in events if e["mode"] == "1:N_register"]
        assert reg_events, "Expected 1:N_register events"
        re0 = reg_events[0]
        print(
            f"[7d] Latest 1:N register latency: backend={re0['search_backend']}, "
            f"candidates={re0['candidate_count']}, search_ms={re0['comparison_latency_ms']:.2f} ms, "
            f"similarity={re0['similarity_score']:.4f}, total_ms={re0['total_latency_ms']:.1f} ms"
        )

        print("\n=== ALL 7 ANTI-JOKI E2E TESTS PASSED! ===")
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)
        if orig_require_pad is None:
            os.environ.pop("REQUIRE_ANTISPOOF", None)
        else:
            os.environ["REQUIRE_ANTISPOOF"] = orig_require_pad
        _cleanup_test_records()
        print("[CLEANUP] Semua data uji TESTJOKI_* berhasil dibersihkan dari PostgreSQL.")


if __name__ == "__main__":
    run_e2e_test()
