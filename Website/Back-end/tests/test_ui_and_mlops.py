from __future__ import annotations
import json
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Face Recognition" / "src"))

from app.main import app
from app.database import SessionLocal
from app.models import DuplicateIdentityAlert, VerificationEvent

PORT = 18766
BASE = f"http://127.0.0.1:{PORT}"


def get(path: str, headers: dict | None = None):
    req = urllib.request.Request(BASE + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def post_mp(path: str, fields: dict, fname: str, fbytes: bytes):
    b = "----TAv2Smoke" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{fname}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
        + fbytes
        + b"\r\n"
    )
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(
        BASE + path,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def main():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(120):
        try:
            c, raw = get("/api/health")
            if c == 200 and json.loads(raw)["model_ready"]:
                break
        except Exception:
            pass
        time.sleep(0.5)

    try:
        for p in ["/", "/static/styles.css", "/static/app.js", "/api/health"]:
            c, raw = get(p)
            print(f"[GET {p}] -> HTTP {c} ({len(raw)} bytes)")
            assert c == 200

        hdr = {"X-Admin-Key": "1234"}
        c, raw = get("/api/admin/stats", hdr)
        stats = json.loads(raw)
        print(
            f"[STATS] users={stats['users']['total']}, vectors={stats['vector_index']['indexed_vectors']}, "
            f"pad_th={stats['model']['pad_live_threshold']}"
        )
        assert stats["model"]["pad_live_threshold"] == "0.430230"

        c, raw = get("/api/admin/mlops", hdr)
        mlops = json.loads(raw)
        print(
            f"[MLOPS] drift={mlops['drift_monitor']['status']}, epochs={len(mlops['training_history'])}, "
            f"dvc={len(mlops['dvc_datasets'])}, "
            f"pad_auc={mlops['pad_threshold_calibration']['test_metrics_at_frozen_validation_threshold']['ROC_AUC']:.4f}"
        )
        assert len(mlops["training_history"]) == 15
        assert len(mlops["dvc_datasets"]) == 2

        c, raw = get("/api/admin/demo-users", hdr)
        demos = json.loads(raw)
        print(f"[DEMO USERS] count={len(demos)}, first={demos[0]['nim'] if demos else None}")
        assert len(demos) == 10

        # Test real printout spoof image against /api/registration/register with calibrated PAD!
        df = pd.read_csv(PROJECT_ROOT / "Face Recognition" / "artifacts" / "thresholds" / "pad_test_predictions.csv")
        spoof_row = df[(df["attack_type"] == "printouts") & (df["pad_live_score"] < 0.05)].iloc[0]
        spoof_path = PROJECT_ROOT / "Dataset" / "raw" / "PAD_calibration" / spoof_row["image_path"]
        c, resp = post_mp(
            "/api/registration/register",
            {"nim": "TEST_SPOOF_01", "full_name": "Test Spoof"},
            "spoof.jpg",
            spoof_path.read_bytes(),
        )
        print(f"[SPOOF CHECK] HTTP {c} -> {resp}")
        assert resp["decision_status"] == "REJECTED_SPOOF", resp

        print("\n=== ALL UI + MLOPS + CALIBRATED PAD SMOKE TESTS PASSED! ===")
    finally:
        srv.should_exit = True
        th.join(timeout=5)
        with SessionLocal() as db:
            db.query(DuplicateIdentityAlert).filter(
                DuplicateIdentityAlert.claimed_nim == "TEST_SPOOF_01"
            ).delete(synchronize_session=False)
            db.query(VerificationEvent).filter(
                VerificationEvent.nim_masked == "TE*********01"
            ).delete(synchronize_session=False)
            db.commit()


if __name__ == "__main__":
    main()
