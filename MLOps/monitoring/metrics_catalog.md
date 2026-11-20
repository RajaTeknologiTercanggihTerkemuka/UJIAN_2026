# Katalog metrik

| Metrik | Definisi | Tampilan |
|---|---|---|
| Detection confidence | Keyakinan MTCNN mendeteksi wajah, bukan identitas | Operator/debug |
| Face count | Jumlah wajah dalam frame | Peserta (instruksi), operator |
| Face quality flags | Blur/brightness/size/pose; awalnya warning | Peserta/operator |
| Similarity score | Cosine similarity template-probe | Operator terbatas |
| Threshold | Batas keputusan untuk model+preprocessing tertentu | Operator terbatas |
| Decision | MATCH/MISMATCH/NO_FACE/MULTIPLE_FACES/QUALITY_RETRY/ERROR | Peserta (status sederhana) + operator |
| Detection/embedding/comparison latency | Durasi komponen dalam ms | Operator/monitoring |
| Total latency | End-to-end dari API menerima gambar sampai keputusan | Dashboard |
| Model/preprocessing/threshold version | Identifikasi pipeline yang menghasilkan skor | Operator/log/MLflow |
| FAR/FRR/EER/ROC-AUC/F1 | Evaluasi dengan ground truth | Laporan penelitian + MLflow |

## Face quality tidak terlalu sensitif
Hard reject awal hanya untuk tidak ada wajah, lebih dari satu wajah pada 1:1, crop invalid, wajah sangat kecil/terpotong. Blur, brightness, pose sebagai warning dulu. Ambang ditentukan melalui uji webcam nyata, bukan ditebak.

## Anti-spoofing
MTCNN dan ArcFace bukan liveness/PAD. Anti-spoofing adalah modul terpisah dengan data serangan print/replay dan evaluasi tersendiri.

## Privasi
Jangan log foto, frame video, embedding mentah, nama/NIM, atau token bila tidak diperlukan. Gunakan request ID acak dan batasi akses dashboard.
