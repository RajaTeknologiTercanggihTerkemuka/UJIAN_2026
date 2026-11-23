from __future__ import annotations
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = Path(__file__).resolve().parent / "demo_seed.sql"


def _sql_str(val: str) -> str:
    return "'" + val.replace("'", "''") + "'"


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise SystemExit("[ERROR] DATABASE_URL tidak ditemukan pada .env")

    eng = create_engine(db_url, pool_pre_ping=True)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT u.nim, u.full_name, u.exam_year, u.status,
                       ft.embedding, ft.model_version, ft.preprocessing_version, ft.source_type
                FROM users u
                JOIN face_templates ft ON ft.user_id = u.id
                WHERE ft.is_active = TRUE
                  AND ft.source_type = 'demo_seed_reference'
                ORDER BY u.nim ASC
                """
            )
        ).fetchall()

    if not rows:
        raise SystemExit("[ERROR] Tidak ada akun demo (source_type='demo_seed_reference') ditemukan di PostgreSQL.")

    lines = [
        "-- Seed 10 akun demo (04221001..04221010) beserta 512-d face embeddings untuk VPS.",
        "-- Jalankan di VPS setelah container db & web menyala:",
        "--   docker compose exec -T db psql -U face_app_user -d face_registration < deploy/demo_seed.sql",
        "BEGIN;",
        "",
    ]

    for nim, full_name, exam_year, status, emb, model_ver, prep_ver, src_type in rows:
        emb_json = json.dumps(emb)
        lines.append(
            f"INSERT INTO users (nim, full_name, exam_year, status) "
            f"VALUES ({_sql_str(nim)}, {_sql_str(full_name)}, {int(exam_year)}, {_sql_str(status)}) "
            f"ON CONFLICT (nim) DO UPDATE SET full_name = EXCLUDED.full_name, exam_year = EXCLUDED.exam_year, status = EXCLUDED.status;"
        )
        lines.append(
            f"DELETE FROM face_templates WHERE user_id = (SELECT id FROM users WHERE nim = {_sql_str(nim)}) "
            f"AND model_version = {_sql_str(model_ver)} AND preprocessing_version = {_sql_str(prep_ver)};"
        )
        lines.append(
            f"INSERT INTO face_templates (user_id, embedding, model_version, preprocessing_version, source_type, is_active) "
            f"SELECT id, {_sql_str(emb_json)}::json, {_sql_str(model_ver)}, {_sql_str(prep_ver)}, {_sql_str(src_type)}, TRUE "
            f"FROM users WHERE nim = {_sql_str(nim)};"
        )
        lines.append("")

    lines.extend(
        [
            "-- Sinkronkan kolom embedding_vec (pgvector HNSW) jika ekstensi pgvector aktif",
            "DO $$",
            "BEGIN",
            "  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='face_templates' AND column_name='embedding_vec') THEN",
            "    UPDATE face_templates SET embedding_vec = (embedding::text)::vector(512) WHERE embedding_vec IS NULL;",
            "  END IF;",
            "END $$;",
            "",
            "COMMIT;",
            "",
        ]
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(f"[OK] Berhasil export {len(rows)} akun demo ke {OUT_PATH} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
