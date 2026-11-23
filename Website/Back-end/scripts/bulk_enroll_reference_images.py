"""Admin batch-enroll reference photos stored either as <IMAGE_ROOT>/<NIM>/*.(jpg|png)
or flat files inside <IMAGE_ROOT> mapped via masterdata CSV / <NIM>_<Nama>.jpeg.
Supports automatic resume by skipping participants already enrolled for the active model.
"""
import argparse, gc, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'Face Recognition' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceid.config import FaceConfig
from faceid.engine import FaceEngine
from app.database import Base, engine, SessionLocal
from app.models import User, FaceTemplate

p = argparse.ArgumentParser()
p.add_argument('--image-root', required=True)
p.add_argument('--nim-column', default='NIM')
p.add_argument('--name-column', default='Nama')
p.add_argument('--photo-column', default='File_Foto')
p.add_argument('--masterdata-csv', required=True)
p.add_argument('--dry-run', action='store_true')
p.add_argument('--overwrite', action='store_true', help='Re-process users that already have templates')
p.add_argument('--max-images-per-person', type=int, default=0)
p.add_argument('--limit-users', type=int, default=0)
a = p.parse_args()

master = pd.read_csv(a.masterdata_csv, dtype=str).fillna('')
if a.nim_column not in master or a.name_column not in master:
    raise SystemExit(f'Kolom NIM/nama tidak ditemukan. Kolom tersedia: {list(master.columns)}')

name_map = dict(zip(master[a.nim_column].str.strip(), master[a.name_column].str.strip()))
photo_to_nim = {}
if a.photo_column in master.columns:
    for nim_val, photo_val in zip(master[a.nim_column].str.strip(), master[a.photo_column].str.strip()):
        if nim_val and photo_val:
            photo_to_nim[Path(photo_val).name] = nim_val

root = Path(a.image_root)
if not root.exists():
    raise SystemExit(f'Image root tidak ditemukan: {root}')

exts = {'.jpg', '.jpeg', '.png'}
grouped_paths = defaultdict(list)
unmapped_files = []

for person_dir in sorted(x for x in root.iterdir() if x.is_dir()):
    nim = person_dir.name.strip()
    paths = sorted(x for x in person_dir.rglob('*') if x.is_file() and x.suffix.lower() in exts)
    if paths:
        grouped_paths[nim].extend(paths)

for img_path in sorted(x for x in root.iterdir() if x.is_file() and x.suffix.lower() in exts):
    nim = photo_to_nim.get(img_path.name) or img_path.stem.split('_')[0].strip()
    if nim:
        grouped_paths[nim].append(img_path)
    else:
        unmapped_files.append(img_path.name)

target_nims = sorted(grouped_paths.keys())
if a.limit_users > 0:
    target_nims = target_nims[:a.limit_users]

config = FaceConfig.from_env()
Base.metadata.create_all(bind=engine)
db = SessionLocal()

existing_nims = set()
if not a.dry_run and not a.overwrite:
    existing_nims = {
        row[0] for row in (
            db.query(User.nim)
            .join(FaceTemplate, FaceTemplate.user_id == User.id)
            .filter(
                FaceTemplate.model_version == config.model_version,
                FaceTemplate.preprocessing_version == config.preprocessing_version,
                FaceTemplate.source_type == 'bulk_reference_images'
            )
            .all()
        )
    }

print(f'Total identitas di folder foto: {len(target_nims)} | Sudah tersimpan di DB (di-skip): {len(existing_nims)}')

face_engine = FaceEngine(config)
total = 0
processed_new = 0
failed = [(f, 'unmapped_filename') for f in unmapped_files]

try:
    for idx, nim in enumerate(target_nims, start=1):
        if nim in existing_nims:
            continue

        if nim not in name_map or not name_map[nim]:
            failed.append((nim, 'not_in_masterdata'))
            continue

        vectors = []
        person_paths = grouped_paths[nim]
        if a.max_images_per_person > 0:
            person_paths = person_paths[:a.max_images_per_person]

        for path in person_paths:
            try:
                result = face_engine.embed_image_bytes(path.read_bytes())
                vectors.append(result['embedding'])
            except Exception as exc:
                failed.append((str(path), getattr(exc, 'code', type(exc).__name__)))

        processed_new += 1
        if processed_new % 100 == 0:
            gc.collect()

        if not vectors:
            continue

        template = np.mean(np.stack(vectors), axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)

        if a.dry_run:
            print(f'[DRY RUN] {nim} ({name_map[nim]}): {len(vectors)} usable reference image(s)')
            total += 1
            continue

        user = db.query(User).filter(User.nim == nim).first()
        if user is None:
            user = User(nim=nim, full_name=name_map[nim], status='active')
            db.add(user)
            db.flush()
        else:
            user.full_name = name_map[nim]
            user.status = 'active'

        (db.query(FaceTemplate)
           .filter(FaceTemplate.user_id == user.id,
                   FaceTemplate.model_version == config.model_version,
                   FaceTemplate.preprocessing_version == config.preprocessing_version,
                   FaceTemplate.source_type == 'bulk_reference_images')
           .delete(synchronize_session=False))

        db.add(FaceTemplate(
            user_id=user.id,
            embedding=template.astype(float).tolist(),
            model_version=config.model_version,
            preprocessing_version=config.preprocessing_version,
            source_type='bulk_reference_images'
        ))
        db.commit()
        total += 1

        if processed_new % 250 == 0:
            print(f'[PROGRESS] Posisi {idx}/{len(target_nims)} | Baru berhasil: {total} | Total di DB: {len(existing_nims) + total} | Gagal/Skip: {len(failed)}')
finally:
    if db is not None:
        db.close()

print(f'Selesai! Baru di-enroll: {total} | Total template di DB: {len(existing_nims) + total} | Gagal/skipped: {len(failed)}')
if failed:
    report = ROOT / 'MLOps' / 'private_bulk_enroll_failures.csv'
    pd.DataFrame(failed, columns=['item', 'reason']).to_csv(report, index=False)
    print('Private failure report:', report)
