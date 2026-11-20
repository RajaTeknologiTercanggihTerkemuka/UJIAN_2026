"""Inspect a .pt registry metadata without displaying raw names/embeddings.
Only load trusted .pt files; torch pickle files can execute code if malicious.
"""
import argparse, torch
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--path', required=True)
args = p.parse_args()
obj = torch.load(Path(args.path), map_location='cpu', weights_only=False)
print('Top-level type:', type(obj).__name__)
if isinstance(obj, dict):
    print('Top-level keys:', list(obj.keys()))
    for key, value in obj.items():
        shape = getattr(value, 'shape', None)
        print(f'{key}: type={type(value).__name__}, shape={shape}')
else:
    print('Shape:', getattr(obj, 'shape', None))
print('Do not mix legacy VGGFace2 embeddings with embeddings from a fine-tuned model.')
