from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "sql" / "migrations"


def split_sql_statements(sql_text: str) -> list[str]:
    """Split a migration SQL script into executable statements, ignoring comments."""
    lines = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    return [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]


def run_sql_file(engine, sql_path: Path) -> None:
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = split_sql_statements(sql_text)
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"[OK] Applied migration: {sql_path.name} ({len(statements)} statements)")


def check_pgvector_available(engine) -> tuple[bool, bool]:
    """Returns (available_on_server, installed_in_db)."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'vector'")
        ).fetchone()
    if row is None:
        return False, False
    return True, (row[1] is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply TA v2 PostgreSQL schema migrations safely.")
    parser.add_argument(
        "--with-pgvector",
        action="store_true",
        help="Also attempt 003_enable_pgvector_hnsw.sql (auto-attempted if pgvector is available).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("[ERROR] DATABASE_URL tidak ditemukan pada .env", file=sys.stderr)
        return 1

    engine = create_engine(database_url, pool_pre_ping=True)

    # Always apply core non-destructive migrations
    for name in ("001_add_antispoof_latency.sql", "002_antijoki_schema.sql"):
        path = MIGRATIONS_DIR / name
        if path.exists():
            run_sql_file(engine, path)

    # Check pgvector availability on PostgreSQL
    if engine.dialect.name == "postgresql":
        available, installed = check_pgvector_available(engine)
        if available:
            print(f"[INFO] pgvector tersedia di PostgreSQL server (installed={installed}). Menjalankan 003_enable_pgvector_hnsw.sql ...")
            try:
                run_sql_file(engine, MIGRATIONS_DIR / "003_enable_pgvector_hnsw.sql")
            except Exception as exc:
                print(
                    f"[WARN] Gagal mengaktifkan pgvector (kemungkinan perlu hak superuser untuk CREATE EXTENSION vector): {exc}\n"
                    "       Jalankan manual sebagai postgres superuser: psql -d face_registration -f Website/Back-end/sql/migrations/003_enable_pgvector_hnsw.sql"
                )
        else:
            msg = (
                "[INFO] Ekstensi pgvector belum terpasang pada PostgreSQL server lokal ini.\n"
                "       Backend otomatis menggunakan fallback high-speed 'numpy_matrix_cache' untuk pencarian 1:N (<5 ms).\n"
                "       Saat deploy VPS Linux (pgvector/pgvector), jalankan 003_enable_pgvector_hnsw.sql untuk mengaktifkan indeks HNSW native."
            )
            print(msg)
            if args.with_pgvector:
                return 2

    print("[DONE] Semua migrasi kompatibel selesai diterapkan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
