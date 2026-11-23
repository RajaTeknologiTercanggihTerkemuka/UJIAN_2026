from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Face Recognition" / "src"))

from app.database import SessionLocal, Base, engine
from app.models import DuplicateIdentityAlert, FaceTemplate, User, VerificationEvent
from app.services.face_service import deactivate_user_templates
from app.services.vector_search import VectorSearchIndex
from faceid.config import FaceConfig
from faceid.engine import FaceEngine

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEMO_NIM_PREFIX = "DEMO2026"
MANIFEST_PATH = Path(__file__).resolve().parent / "demo_identities_manifest.json"


def reset_demo_data() -> None:
    with SessionLocal() as db:
        demo_users = db.query(User).filter(User.nim.like(f"{DEMO_NIM_PREFIX}%")).all()
        demo_ids = [u.id for u in demo_users]
        if demo_ids:
            db.query(DuplicateIdentityAlert).filter(
                (DuplicateIdentityAlert.claimed_nim.like(f"{DEMO_NIM_PREFIX}%"))
                | (DuplicateIdentityAlert.matched_user_id.in_(demo_ids))
            ).delete(synchronize_session=False)
            db.query(VerificationEvent).filter(
                (VerificationEvent.user_id.in_(demo_ids))
                | (VerificationEvent.matched_user_id.in_(demo_ids))
            ).delete(synchronize_session=False)
            db.query(FaceTemplate).filter(FaceTemplate.user_id.in_(demo_ids)).delete(synchronize_session=False)
            db.query(User).filter(User.id.in_(demo_ids)).delete(synchronize_session=False)
        else:
            db.query(DuplicateIdentityAlert).filter(
                DuplicateIdentityAlert.claimed_nim.like(f"{DEMO_NIM_PREFIX}%")
            ).delete(synchronize_session=False)
        db.commit()
    print(f"[OK] Dihapus {len(demo_ids)} akun demo ({DEMO_NIM_PREFIX}*). Data non-demo tidak disentuh.")


def find_candidate_folders(source_dir: Path, min_images: int = 2) -> list[tuple[Path, list[Path]]]:
    candidates: list[tuple[Path, list[Path]]] = []
    if not source_dir.exists():
        return candidates
    for sub in sorted(source_dir.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) >= min_images:
            candidates.append((sub, imgs))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed 10 clean demo identities (DEMO20260001..DEMO20260010).")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "raw" / "IMSFD_original" / "A1" / "Testing",
        help="Folder containing one subfolder per identity (>=2 images per folder).",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of demo identities to seed.")
    parser.add_argument("--reset", action="store_true", help="Remove existing DEMO2026* records before seeding (or only reset with --only-reset).")
    parser.add_argument("--only-reset", action="store_true", help="Only delete DEMO2026* records and exit.")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    if args.only_reset:
        reset_demo_data()
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        return 0

    if args.reset:
        reset_demo_data()

    candidates = find_candidate_folders(args.source_dir, min_images=2)
    if len(candidates) < args.count:
        print(f"[ERROR] Hanya ditemukan {len(candidates)} folder identitas valid di {args.source_dir} (butuh {args.count}).")
        return 1

    cfg = FaceConfig.from_env()
    face_engine = FaceEngine(cfg)
    vector_index = VectorSearchIndex(cfg.model_version, cfg.preprocessing_version)

    manifest_entries = []
    enrolled = 0

    with SessionLocal() as db:
        vector_index.load_from_db(db)
        for folder, imgs in candidates:
            if enrolled >= args.count:
                break
            enroll_img = None
            test_img = None
            embed_res = None

            # Pick first valid single-face image (non-colliding with existing gallery) for enrollment
            # and a distinct image of the same identity for demo probe testing.
            dedup_th = cfg.dedup_threshold or 0.75
            for img_path in imgs[:8]:
                try:
                    res = face_engine.embed_image_bytes(img_path.read_bytes())
                except Exception:
                    continue
                if enroll_img is None:
                    check = vector_index.search(db, res["embedding"], top_k=1)
                    if check.matches and check.matches[0].similarity >= dedup_th:
                        break  # Skip folder if lookalike already exists in gallery
                    enroll_img = img_path
                    embed_res = res
                else:
                    test_img = img_path
                    break

            if enroll_img is None or test_img is None or embed_res is None:
                continue

            enrolled += 1
            demo_nim = f"{DEMO_NIM_PREFIX}{enrolled:04d}"
            demo_name = f"Peserta Demo {enrolled:02d}"

            user = db.query(User).filter(User.nim == demo_nim).first()
            if user is None:
                user = User(nim=demo_nim, full_name=demo_name, exam_year=2026, status="active")
                db.add(user)
                db.flush()
            else:
                user.full_name = demo_name
                user.exam_year = 2026
                user.status = "active"
                user.updated_at = datetime.now(timezone.utc)

            deactivate_user_templates(db, user.id, cfg.model_version, cfg.preprocessing_version)
            tpl = FaceTemplate(
                user_id=user.id,
                embedding=embed_res["embedding"].astype(float).tolist(),
                model_version=cfg.model_version,
                preprocessing_version=cfg.preprocessing_version,
                source_type="demo_seed_reference",
                is_active=True,
            )
            db.add(tpl)
            db.commit()
            db.refresh(tpl)
            vector_index.sync_new_template(
                db, tpl.id, user.id, user.nim, user.full_name, user.status, embed_res["embedding"]
            )

            manifest_entries.append(
                {
                    "demo_nim": demo_nim,
                    "demo_name": demo_name,
                    "exam_year": 2026,
                    "enrolled_image": str(enroll_img.relative_to(PROJECT_ROOT)),
                    "probe_test_image": str(test_img.relative_to(PROJECT_ROOT)),
                    "detection_confidence": round(float(embed_res["detection_confidence"]), 4),
                }
            )
            print(f"[SEED {enrolled:02d}/{args.count}] {demo_nim} ({demo_name}) -> OK (conf={embed_res['detection_confidence']:.4f})")

    MANIFEST_PATH.write_text(json.dumps({"model_version": cfg.model_version, "identities": manifest_entries}, indent=2), encoding="utf-8")
    print(f"[DONE] {enrolled} identitas demo tersimpan di PostgreSQL. Manifest uji: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
