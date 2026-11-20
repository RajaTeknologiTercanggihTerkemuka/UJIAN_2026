from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Face Recognition" / "src"))
load_dotenv(ROOT / ".env")

from faceid.config import FaceConfig
from faceid.engine import FaceEngine

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
args = parser.parse_args()
config = FaceConfig.from_env()
engine = FaceEngine(config)
result = engine.embed_image_bytes(Path(args.image).read_bytes())
print(json.dumps({
    'embedding_dim': int(result['embedding'].shape[0]),
    'embedding_norm': float((result['embedding'] ** 2).sum() ** 0.5),
    'detection_confidence': result['detection_confidence'],
    'face_count': result['face_count'],
    'quality': result['quality'],
    'detection_latency_ms': result['detection_latency_ms'],
    'embedding_latency_ms': result['embedding_latency_ms'],
    'model_version': config.model_version,
    'preprocessing_version': config.preprocessing_version,
}, indent=2))
