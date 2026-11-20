from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from facenet_pytorch import MTCNN

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Face Recognition' / 'src'))
from faceid.antispoof.silentface_provider import SilentFaceMiniFASNetProvider
from faceid.antispoof.model_loader_device import choose_device_for_pad


def main():
    parser = argparse.ArgumentParser(description='Predict PAD scores for a labelled camera-capture manifest.')
    parser.add_argument('--manifest', required=True, help='CSV with image_path and label columns')
    parser.add_argument('--image-column', default='image_path')
    parser.add_argument('--label-column', default='label', help='1=bona fide, 0=attack')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    df = pd.read_csv(manifest_path)
    for col in [args.image_column, args.label_column]:
        if col not in df.columns:
            raise ValueError(f'Kolom {col!r} tidak ada pada manifest')
    df[args.label_column] = pd.to_numeric(df[args.label_column], errors='raise').astype(int)
    if not set(df[args.label_column].unique()).issubset({0, 1}):
        raise ValueError('Label harus 1=bona fide dan 0=attack')

    device = choose_device_for_pad(os.getenv('FACE_DEVICE', 'auto'))
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, post_process=False, device=device).eval()
    weights_dir = Path(os.getenv('PAD_WEIGHTS_DIR', ROOT / 'Face Recognition/artifacts/antispoof/weights'))
    repo_root = Path(os.getenv('PAD_REPO_ROOT', ROOT / 'Face Recognition/third_party/Silent-Face-Anti-Spoofing'))
    if not weights_dir.is_absolute(): weights_dir = ROOT / weights_dir
    if not repo_root.is_absolute(): repo_root = ROOT / repo_root
    provider = SilentFaceMiniFASNetProvider(weights_dir, repo_root, device, live_threshold=None)

    out_rows = []
    for row in df.to_dict('records'):
        raw_path = Path(str(row[args.image_column]))
        path = raw_path if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
        item = dict(row)
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                arr = np.asarray(image, dtype=np.uint8)
            boxes, probs = mtcnn.detect(Image.fromarray(arr))
            if boxes is None or len(boxes) != 1:
                item.update({'pad_live_score': None, 'pad_status': 'invalid_face_count', 'error': f'face_count={0 if boxes is None else len(boxes)}'})
            else:
                x1, y1, x2, y2 = [float(v) for v in boxes[0]]
                result = provider.predict(arr, [x1, y1, x2-x1, y2-y1])
                item.update({'pad_live_score': result.score, 'pad_status': result.status, 'error': result.reason})
        except Exception as exc:
            item.update({'pad_live_score': None, 'pad_status': 'error', 'error': repr(exc)})
        out_rows.append(item)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output_path, index=False)
    print('PAD prediction CSV:', output_path)
    print(pd.DataFrame(out_rows)['pad_status'].value_counts(dropna=False))
    print('Interpretasikan label dan skor hanya setelah memeriksa protokol capture dan asal sampelnya.')

if __name__ == '__main__':
    main()
