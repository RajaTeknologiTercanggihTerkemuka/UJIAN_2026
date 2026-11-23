# Laporan Hasil Eksperimen & Pengujian Sistem — TA v2

Semua data pada dokumen ini dihasilkan secara langsung dari pengujian nyata terhadap model terlatih (`imsfd_partial_ft_v1`), kalibrasi MTCNN & PAD (`MiniFASNetV2`), database PostgreSQL (`12.886+` vektor), dan server VPS Hostinger (`https://ujian.153.92.5.145.sslip.io`).

---

## 1. Perbandingan Pretrained VGGFace2 vs Partial Fine-Tuning IMSFD
| model | threshold | accuracy | precision | recall | f1_score | FAR | FRR | ROC_AUC | test_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline Pretrained VGGFace2 | 0.5991 | 0.9531 | 0.9532 | 0.953 | 0.9531 | 0.0468 | 0.047 | 0.9738 | 26460 |
| Partial Fine-Tuned IMSFD (ArcFace) | 0.323 | 0.9925 | 0.993 | 0.992 | 0.9925 | 0.007 | 0.008 | 0.9977 | 26460 |

- **Kesimpulan:** Partial fine-tuning pada IMSFD meningkatkan Accuracy dari **95,31% → 99,25%**, ROC-AUC dari **97,38% → 99,77%**, dan memangkas **FAR hingga 6,7× lebih rendah** (`4,68% → 0,70%`).
- Grafik: `fig1_pretrained_vs_partial_ft.png`

---

## 2. Threshold Testing (FAR, FRR, EER & Pemisahan 1:1 vs 1:N)
| target_threshold | actual_threshold | FAR_pct | FRR_pct | role |
| --- | --- | --- | --- | --- |
| 0.25 | 0.2502 | 4.453 | 0.377 | Titik Uji |
| 0.3 | 0.3002 | 1.555 | 0.423 | Titik Uji |
| 0.3651 | 0.3651 | 0.423 | 0.423 | Optimal EER (Aktif 1:1 Verify) |
| 0.45 | 0.4423 | 0.03 | 0.468 | Titik Uji |
| 0.5519 | 0.5519 | 0.0 | 0.694 | FAR=0% Validasi IMSFD |
| 0.65 | 0.6504 | 0.0 | 1.525 | Titik Uji |
| 0.75 | 0.7504 | 0.0 | 4.362 | Aktif 1:N Anti-Joki (12.886 galeri) |

- Grafik: `fig2_threshold_far_frr_curve.png`

---

## 3. Pengujian Anti-Spoofing / PAD (`MiniFASNetV2`, Threshold = `0.4302`)
- Aggregate Test Set (`n=389`): **ROC-AUC = 95,10%**, **BPCER = 1,98%**, **APCER = 21,88%**, **ACER = 11,93%**.

| attack_type | class | samples | mean_live_score | median_live_score | error_metric | error_rate_pct | correct_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cut-out printouts | Attack (Spoof) | 100 | 0.4036 | 0.2443 | APCER | 42.0 | 58.0 |
| live_selfie | Bona Fide (Live) | 4 | 0.8807 | 0.8816 | BPCER | 0.0 | 100.0 |
| live_video | Bona Fide (Live) | 97 | 0.9203 | 0.9929 | BPCER | 2.06 | 97.94 |
| printouts | Attack (Spoof) | 93 | 0.05 | 0.0001 | APCER | 4.3 | 95.7 |
| replay | Attack (Spoof) | 95 | 0.1641 | 0.001 | APCER | 17.89 | 82.11 |

- Grafik & Confusion Matrix: `fig3_pad_evaluation.png`

---

## 4. Pengujian Joki Tipe A (Penggantian Orang 1:1 & Presentation Attack)
| scenario | total | passed | avg_sim | avg_pad | rate_pct |
| --- | --- | --- | --- | --- | --- |
| Joki Tipe A (Penggantian Orang 1:1) | 14 | 14 | 0.0198 | 0.99 | 100.0 |
| Joki Tipe A Spoof (printouts) | 9 | 7 | 0.0832 | 0.0003 | 77.78 |
| Joki Tipe A Spoof (replay) | 11 | 11 | -0.0343 | 0.0445 | 100.0 |
| Verifikasi Sah (Genuine) | 14 | 14 | 0.8938 | 0.99 | 100.0 |

---

## 5. Pengujian Joki Tipe B (Manipulasi Enrollment / Duplikasi 1:N pada 12.886+ Vektor)
| scenario | total | passed | avg_sim | avg_search_ms | rate_pct |
| --- | --- | --- | --- | --- | --- |
| Daftar Ulang Sah (Wajah & NIM Sama) | 14 | 12 | 0.8938 | 1.75 | 85.71 |
| Joki Tipe B (Wajah Sama NIM Beda) | 14 | 12 | 0.8567 | 1.76 | 85.71 |
| Manipulasi NIM Terdaftar | 14 | 14 | 1.0 | 1.77 | 100.0 |
| Pendaftar Baru Sah | 14 | 14 | 0.6608 | 1.72 | 100.0 |

- Grafik gabungan Anti-Joki A & B: `fig4_joki_detection_rates.png`

---

## 6. Pengujian HNSW (`pgvector` pada VPS Hostinger)
Jalankan skrip siap pakai `deploy/benchmark_hnsw_vps.py` di PuTTY VPS:
```bash
docker compose exec web python /app/deploy/benchmark_hnsw_vps.py
```
Skrip mengukur **Recall@1, Recall@5, Mean/p50/p95 Latency** membandingkan Exact Scan vs `pgvector HNSW` (`ef_search = 10, 40, 100`) pada 5.000 vektor 512-d.

---

## 7. Simulasi Drift Monitoring (Normal vs Degradasi Cahaya/Blur)
| condition | brightness_factor | gaussian_blur_radius | detection_rate_pct | avg_genuine_similarity | simulated_failure_rate_pct | drift_monitor_status |
| --- | --- | --- | --- | --- | --- | --- |
| Normal (Baseline) | 1.0 | 0.0 | 100.0 | 0.8647 | 5.0 | HEALTHY |
| Drift Ringan (Gelap + Blur Ringan) | 0.4 | 2.2 | 100.0 | 0.8577 | 30.0 | DRIFT_WARNING |
| Drift Berat (Sangat Gelap + Blur Kuat) | 0.22 | 4.2 | 100.0 | 0.7856 | 55.0 | RETRAINING_RECOMMENDED |

- Grafik: `fig6_drift_simulation.png`

---

## 8. Load Testing Server VPS Hostinger (`https://ujian.153.92.5.145.sslip.io`)
| endpoint | concurrency | total_requests | success_rate_pct | throughput_rps | p50_ms | p95_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GET /api/health | 1 | 10 | 100.0 | 3.63 | 236.3 | 454.2 | 581.3 |
| GET /api/health | 5 | 20 | 100.0 | 19.82 | 246.0 | 291.0 | 297.9 |
| GET /api/health | 10 | 30 | 100.0 | 36.53 | 262.9 | 288.5 | 293.0 |
| POST /api/verification/verify (Full AI) | 1 | 5 | 100.0 | 2.83 | 358.5 | 386.5 | 387.1 |
| POST /api/verification/verify (Full AI) | 3 | 9 | 100.0 | 7.56 | 346.5 | 495.5 | 498.1 |
| POST /api/verification/verify (Full AI) | 5 | 10 | 100.0 | 13.52 | 360.4 | 375.6 | 380.9 |

- Grafik: `fig7_load_test.png`

---

## 9. End-to-End & Edge-Case Black-Box Testing (12 Skenario)
| test_id | scenario | expected | actual | status | latency_ms |
| --- | --- | --- | --- | --- | --- |
| TC-01 | Health Check API | HTTP 200 & model_ready=True | HTTP 200, ready=True | PASS | 19.5 |
| TC-02 | Admin Tanpa Password (Security) | HTTP 401 Unauthorized | HTTP 401 | PASS | 29.4 |
| TC-03 | Admin Login Valid (1234) | HTTP 200 | HTTP 200 (851B) | PASS | 52.1 |
| TC-04 | MLOps & Drift Panel API | HTTP 200 | HTTP 200 (4512B) | PASS | 1259.6 |
| TC-05 | Prometheus /metrics Endpoint | HTTP 200 | HTTP 200 (6342B) | PASS | 16.0 |
| TC-06 | Validasi NIM < 8 Digit | HTTP 422 | HTTP 422 (NIM harus terdiri dari 8 hingga 9 digit angka (contoh: 04221001).) | PASS | 92.2 |
| TC-07 | Kamera Kosong (No Face) | HTTP 422 no_face | HTTP 422 (no_face) | PASS | 50.5 |
| TC-08 | Dua Wajah Sekaligus (Multiple Faces) | HTTP 422 multiple_faces | HTTP 422 (multiple_faces) | PASS | 102.6 |
| TC-09 | Active Liveness Gagal pada Wajah Statis | REJECTED_SPOOF | REJECTED_SPOOF | PASS | 764.2 |
| TC-10 | Serangan Foto Cetak (PAD MiniFASNetV2) | REJECTED_SPOOF | REJECTED_SPOOF | PASS | 760.4 |
| TC-11 | Verifikasi Nama Tidak Cocok NIM | HTTP 422 | HTTP 422 (Nama tidak sesuai dengan NIM yang terdaftar.) | PASS | 27.8 |
| TC-12 | Joki Tipe B (HTTP Duplikat Wajah 1:N) | REJECTED_DUPLICATE_FACE | REJECTED_DUPLICATE_FACE | PASS | 720.3 |

---

## 10. Breakdown Latency Pipeline per Tahap (CPU)
| stage | mean_ms | p95_ms |
| --- | --- | --- |
| 1. Deteksi & Crop Wajah (MTCNN) | 613.13 | 2116.11 |
| 2. Anti-Spoofing PAD (MiniFASNetV2) | 31.15 | 68.0 |
| 3. Ekstraksi Embedding (InceptionResNetV1) | 89.47 | 118.28 |
| 4. Pencarian Vektor 1:N (12.886+ wajah) | 1.75 | 1.98 |

- Grafik: `fig5_latency_breakdown.png`
