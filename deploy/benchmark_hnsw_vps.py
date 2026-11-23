"""
Benchmark pgvector HNSW vs Exact Sequential Scan directly on PostgreSQL 17 (pgvector).
Safe: creates a TEMPORARY table (_bench_hnsw_tmp), runs 50 queries on 5,000 normalized 512-d vectors,
measures Recall@1, Recall@5, and p50/p95 Latency (ms), then drops the temp table automatically.

Run inside Hostinger VPS Docker:
  docker compose exec web python /app/deploy/benchmark_hnsw_vps.py
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
from sqlalchemy import create_engine, text


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL tidak ditemukan.")

    rng = np.random.default_rng(42)
    n_gallery = 5000
    n_queries = 50
    dim = 512

    print(f"[HNSW BENCHMARK] Generating {n_gallery} normalized {dim}-d gallery vectors & {n_queries} queries...")
    gallery = rng.standard_normal((n_gallery, dim)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    # Mix 25 near-duplicate queries (noisy gallery vectors) + 25 random queries
    queries = np.vstack([
        gallery[:25] + rng.normal(0, 0.08, size=(25, dim)).astype(np.float32),
        rng.standard_normal((25, dim)).astype(np.float32),
    ])
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Exact ground truth top-5 via NumPy
    gt_top5 = []
    for q in queries:
        sims = gallery @ q
        top5 = np.argsort(sims)[::-1][:5] + 1  # 1-based SQL ids
        gt_top5.append(top5.tolist())

    eng = create_engine(db_url, pool_pre_ping=True)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("DROP TABLE IF EXISTS _bench_hnsw_tmp"))
        conn.execute(text("CREATE TABLE _bench_hnsw_tmp (id SERIAL PRIMARY KEY, emb vector(512) NOT NULL)"))

        print(f"[HNSW BENCHMARK] Inserting {n_gallery} vectors into temporary table...")
        t_ins = time.perf_counter()
        batch_size = 500
        for start in range(0, n_gallery, batch_size):
            batch = [{"v": fmt_vec(v)} for v in gallery[start:start + batch_size]]
            conn.execute(text("INSERT INTO _bench_hnsw_tmp (emb) VALUES (CAST(:v AS vector))"), batch)
        print(f"  Inserted in {(time.perf_counter() - t_ins):.2f}s")

        results = []

        # 1. Exact Sequential Scan (no index)
        lat_exact = []
        r1_hits, r5_hits = 0, 0
        for qi, q in enumerate(queries):
            qlit = fmt_vec(q)
            t0 = time.perf_counter()
            rows = conn.execute(
                text("SELECT id FROM _bench_hnsw_tmp ORDER BY emb <=> CAST(:q AS vector) LIMIT 5"),
                {"q": qlit},
            ).fetchall()
            lat_exact.append((time.perf_counter() - t0) * 1000.0)
            got = [int(r[0]) for r in rows]
            if got and got[0] == gt_top5[qi][0]:
                r1_hits += 1
            r5_hits += len(set(got) & set(gt_top5[qi]))
        results.append({
            "method": "Exact Scan (No Index)",
            "ef_search": None,
            "recall_at_1": round(r1_hits / n_queries, 4),
            "recall_at_5": round(r5_hits / (n_queries * 5), 4),
            "p50_ms": round(float(np.percentile(lat_exact, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_exact, 95)), 2),
            "mean_ms": round(float(np.mean(lat_exact)), 2),
        })

        # 2. Build HNSW Index (m=16, ef_construction=64)
        print("[HNSW BENCHMARK] Building HNSW index (m=16, ef_construction=64)...")
        t_idx = time.perf_counter()
        conn.execute(text(
            "CREATE INDEX _bench_hnsw_idx ON _bench_hnsw_tmp USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ))
        build_s = time.perf_counter() - t_idx
        print(f"  HNSW index built in {build_s:.2f}s")

        for ef in [10, 40, 100]:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
            lats = []
            r1_hits, r5_hits = 0, 0
            for qi, q in enumerate(queries):
                qlit = fmt_vec(q)
                t0 = time.perf_counter()
                rows = conn.execute(
                    text("SELECT id FROM _bench_hnsw_tmp ORDER BY emb <=> CAST(:q AS vector) LIMIT 5"),
                    {"q": qlit},
                ).fetchall()
                lats.append((time.perf_counter() - t0) * 1000.0)
                got = [int(r[0]) for r in rows]
                if got and got[0] == gt_top5[qi][0]:
                    r1_hits += 1
                r5_hits += len(set(got) & set(gt_top5[qi]))
            results.append({
                "method": f"pgvector HNSW (ef_search={ef})",
                "ef_search": ef,
                "recall_at_1": round(r1_hits / n_queries, 4),
                "recall_at_5": round(r5_hits / (n_queries * 5), 4),
                "p50_ms": round(float(np.percentile(lats, 50)), 2),
                "p95_ms": round(float(np.percentile(lats, 95)), 2),
                "mean_ms": round(float(np.mean(lats)), 2),
            })

        conn.execute(text("DROP TABLE IF EXISTS _bench_hnsw_tmp"))

    print("\n=== HASIL BENCHMARK PGVECTOR HNSW (5.000 VEKTOR 512-D, 50 QUERY) ===")
    print(f"{'Method':32s} | {'Recall@1':>8s} | {'Recall@5':>8s} | {'Mean ms':>8s} | {'p50 ms':>7s} | {'p95 ms':>7s}")
    print("-" * 86)
    for r in results:
        print(
            f"{r['method']:32s} | {r['recall_at_1']*100:7.2f}% | {r['recall_at_5']*100:7.2f}% | "
            f"{r['mean_ms']:8.2f} | {r['p50_ms']:7.2f} | {r['p95_ms']:7.2f}"
        )
    print("\nJSON:", json.dumps({"n_gallery": n_gallery, "n_queries": n_queries, "hnsw_build_sec": round(build_s, 2), "results": results}, indent=2))


if __name__ == "__main__":
    main()
