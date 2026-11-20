"""Calibrate cosine threshold on a validation-only set using the production inference pipeline.
Use either --validation-root/<identity>/*.jpg or --validation-manifest CSV(path, identity).
Never point this script at the final test set.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'Face Recognition' / 'src'))
from faceid.config import FaceConfig
from faceid.engine import FaceEngine, cosine_similarity


def load_groups(args):
    groups = {}
    if args.validation_manifest:
        df = pd.read_csv(args.validation_manifest)
        if not {'path', 'identity'}.issubset(df.columns):
            raise ValueError('CSV validation manifest harus memiliki kolom path dan identity')
        for identity, rows in df.groupby('identity'):
            groups[str(identity)] = [Path(x) for x in rows['path'].astype(str).tolist()]
    else:
        root = Path(args.validation_root)
        if not root.exists():
            raise FileNotFoundError(root)
        groups = {
            d.name: sorted(p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            for d in root.iterdir() if d.is_dir()
        }
    return groups


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--validation-root', help='Folder per identity: root/identity/image.jpg')
    source.add_argument('--validation-manifest', help='CSV with path,identity columns')
    parser.add_argument('--output', default='Face Recognition/artifacts/thresholds/threshold_v1.json')
    parser.add_argument('--pairs-per-identity', type=int, default=100)
    parser.add_argument('--target-far', type=float, default=None, help='Optional policy target, e.g. 0.001. A small validation set may be unable to estimate very low FAR reliably.')
    args = parser.parse_args()
    if args.target_far is not None and not (0 < args.target_far < 1):
        raise SystemExit('--target-far must be between 0 and 1')

    groups = load_groups(args)
    engine = FaceEngine(FaceConfig.from_env())
    identity_vectors = {}
    for identity, image_paths in groups.items():
        vectors = []
        for image_path in image_paths:
            if not image_path.exists():
                print('PATH NOT FOUND:', image_path)
                continue
            try:
                result = engine.embed_image_bytes(image_path.read_bytes())
                vectors.append(result['embedding'])
            except Exception as exc:
                print('SKIP', image_path.name, getattr(exc, 'code', type(exc).__name__))
        if len(vectors) >= 2:
            identity_vectors[identity] = vectors

    if len(identity_vectors) < 2:
        raise SystemExit('Diperlukan minimal dua identitas, masing-masing minimal dua foto validation yang valid.')

    rng = np.random.default_rng(2026)
    genuine_scores, impostor_scores = [], []
    identities = list(identity_vectors)
    for identity, vectors in identity_vectors.items():
        possible = len(vectors) * (len(vectors) - 1) // 2
        pair_count = min(args.pairs_per_identity, possible)
        chosen = set()
        while len(chosen) < pair_count:
            i, j = rng.choice(len(vectors), 2, replace=False)
            chosen.add(tuple(sorted((int(i), int(j)))))
        for i, j in chosen:
            genuine_scores.append(cosine_similarity(vectors[i], vectors[j]))
    for _ in range(len(genuine_scores)):
        id1, id2 = rng.choice(identities, 2, replace=False)
        v1 = identity_vectors[id1][int(rng.integers(len(identity_vectors[id1])))]
        v2 = identity_vectors[id2][int(rng.integers(len(identity_vectors[id2])))]
        impostor_scores.append(cosine_similarity(v1, v2))

    scores = np.asarray(genuine_scores + impostor_scores, dtype=float)
    labels = np.asarray([1] * len(genuine_scores) + [0] * len(impostor_scores), dtype=int)
    thresholds = np.unique(scores)
    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        far = fp / (fp + tn) if fp + tn else 0.0
        frr = fn / (fn + tp) if fn + tp else 0.0
        rows.append((float(threshold), far, frr, abs(far - frr)))

    if args.target_far is None:
        best = min(rows, key=lambda row: (row[3], row[1]))
        selection = 'min_absolute_far_frr_gap_eer_estimate'
    else:
        eligible = [row for row in rows if row[1] <= args.target_far]
        if not eligible:
            raise SystemExit('Tidak ada threshold yang mencapai target FAR pada validation set. Tambah data impostor atau revisi target melalui protokol penelitian.')
        best = min(eligible, key=lambda row: (row[2], -row[0]))
        selection = f'min_frr_subject_to_far<={args.target_far}'

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'threshold': best[0], 'validation_far': best[1], 'validation_frr': best[2],
        'eer_estimate': (best[1] + best[2]) / 2.0 if args.target_far is None else None,
        'target_far': args.target_far,
        'genuine_pairs': len(genuine_scores), 'impostor_pairs': len(impostor_scores),
        'validation_identities': len(identity_vectors),
        'model_version': engine.config.model_version,
        'preprocessing_version': engine.config.preprocessing_version,
        'threshold_version': out.stem, 'selection': selection,
    }
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows, columns=['threshold', 'FAR', 'FRR', 'gap']).to_csv(out.with_suffix('.csv'), index=False)
    print(json.dumps(result, indent=2))
    print('Threshold ini hanya berlaku untuk model/preprocessing/dataset/protokol validasi ini.')

if __name__ == '__main__':
    main()
