import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from faceid.engine import cosine_similarity

def test_identical_vectors_similarity_one():
    x = np.ones(512, dtype=np.float32)
    assert abs(cosine_similarity(x, x) - 1.0) < 1e-6

def test_orthogonal_vectors_similarity_zero():
    a = np.zeros(512, dtype=np.float32); b = np.zeros(512, dtype=np.float32)
    a[0] = 1; b[1] = 1
    assert abs(cosine_similarity(a, b)) < 1e-6
