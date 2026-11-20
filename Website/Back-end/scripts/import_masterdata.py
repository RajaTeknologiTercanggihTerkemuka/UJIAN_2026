"""Import participant metadata only. Does not import/store photos or face embeddings."""
import argparse, os, sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import Base, engine, SessionLocal
from app.models import User

p=argparse.ArgumentParser()
p.add_argument('--csv',required=True)
p.add_argument('--nim-column',default='NIM')
p.add_argument('--name-column',default='Nama')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
df=pd.read_csv(a.csv,dtype=str).fillna('')
for c in [a.nim_column,a.name_column]:
    if c not in df.columns: raise SystemExit(f'Kolom {c!r} tidak ditemukan; kolom tersedia: {list(df.columns)}')
rows=df[[a.nim_column,a.name_column]].copy()
rows[a.nim_column]=rows[a.nim_column].str.strip(); rows[a.name_column]=rows[a.name_column].str.strip()
rows=rows[(rows[a.nim_column]!='') & (rows[a.name_column]!='')].drop_duplicates(a.nim_column)
print('Unique records to import:',len(rows))
if a.dry_run:
    print('Dry-run: tidak ada data dimasukkan.')
    raise SystemExit(0)
Base.metadata.create_all(bind=engine)
db=SessionLocal(); inserted=updated=0
try:
    for r in rows.itertuples(index=False,name=None):
        nim,name=r
        user=db.query(User).filter(User.nim==nim).first()
        if user is None:
            db.add(User(nim=nim,full_name=name,status='active')); inserted+=1
        else:
            user.full_name=name; user.status='active'; updated+=1
    db.commit()
finally:
    db.close()
print('Inserted:',inserted,'Updated:',updated)
print('Do not upload this CSV or database dump into public Git/MLflow.')
