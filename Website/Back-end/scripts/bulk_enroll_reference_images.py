"""Admin batch-enroll reference photos stored as <IMAGE_ROOT>/<NIM>/*.(jpg|png).
This does not create liveness proof; it creates face templates using the active model.
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[3]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT/'Face Recognition'/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from faceid.config import FaceConfig
from faceid.engine import FaceEngine
from app.database import Base, engine, SessionLocal
from app.models import User, FaceTemplate

p=argparse.ArgumentParser()
p.add_argument('--image-root',required=True)
p.add_argument('--nim-column',default='NIM')
p.add_argument('--name-column',default='Nama')
p.add_argument('--masterdata-csv',required=True)
p.add_argument('--dry-run',action='store_true')
p.add_argument('--max-images-per-person', type=int, default=0, help='0=use all images; positive value limits reference photos per participant')
a=p.parse_args()
import pandas as pd
master=pd.read_csv(a.masterdata_csv,dtype=str).fillna('')
if a.nim_column not in master or a.name_column not in master: raise SystemExit('Kolom NIM/nama tidak ditemukan')
name_map=dict(zip(master[a.nim_column].str.strip(), master[a.name_column].str.strip()))
root=Path(a.image_root)
if not root.exists(): raise SystemExit(f'Image root tidak ditemukan: {root}')
config=FaceConfig.from_env(); face_engine=FaceEngine(config)
Base.metadata.create_all(bind=engine)
exts={'.jpg','.jpeg','.png'}
db=SessionLocal(); total=0; failed=[]
try:
    for person_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        nim=person_dir.name
        if nim not in name_map or not name_map[nim]:
            failed.append((nim,'not_in_masterdata')); continue
        vectors=[]
        person_paths = sorted(x for x in person_dir.rglob('*') if x.is_file() and x.suffix.lower() in exts)
        if a.max_images_per_person > 0:
            person_paths = person_paths[:a.max_images_per_person]
        for path in person_paths:
            try:
                result=face_engine.embed_image_bytes(path.read_bytes())
                vectors.append(result['embedding'])
            except Exception as exc:
                failed.append((str(path),getattr(exc,'code',type(exc).__name__)))
        if not vectors: continue
        # Average normalized vectors to create a template; retain a consistent model/preprocess version.
        template=np.mean(np.stack(vectors),axis=0)
        template=template/(np.linalg.norm(template)+1e-12)
        if a.dry_run:
            print(f'[DRY RUN] {nim}: {len(vectors)} usable reference image(s)')
            continue
        user=db.query(User).filter(User.nim==nim).first()
        if user is None:
            user=User(nim=nim,full_name=name_map[nim],status='active'); db.add(user); db.flush()
        else:
            user.full_name=name_map[nim]; user.status='active'
        # Idempotent import: replace existing template for this exact model + preprocessing.
        (db.query(FaceTemplate)
           .filter(FaceTemplate.user_id == user.id,
                   FaceTemplate.model_version == config.model_version,
                   FaceTemplate.preprocessing_version == config.preprocessing_version,
                   FaceTemplate.source_type == 'bulk_reference_images')
           .delete(synchronize_session=False))
        db.add(FaceTemplate(user_id=user.id,embedding=template.astype(float).tolist(),
            model_version=config.model_version,preprocessing_version=config.preprocessing_version,
            source_type='bulk_reference_images'))
        db.commit(); total+=1
finally:
    if db is not None:
        db.close()
print('Users enrolled:',total,'Failed/skipped:',len(failed))
if failed:
    # Only local private report; it may contain names/paths, do not publish.
    report=ROOT/'MLOps'/'private_bulk_enroll_failures.csv'
    pd.DataFrame(failed,columns=['item','reason']).to_csv(report,index=False)
    print('Private failure report:',report)
