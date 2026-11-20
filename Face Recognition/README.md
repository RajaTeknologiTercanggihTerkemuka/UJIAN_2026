# Face Recognition

Modul AI terpisah dari website. Model inti penelitian saat ini adalah `InceptionResnetV1` pretrained VGGFace2 yang diadaptasi melalui partial fine-tuning pada IMSFD dengan ArcFace sebagai objective pelatihan.

## Komponen yang direncanakan

- `detection/`: deteksi wajah, misalnya MTCNN, hanya bila input belum berupa face crop.
- `preprocessing/`: konversi RGB, orientasi, crop/padding, resize 160x160, dan normalisasi yang konsisten.
- `models/`: pembangun InceptionResnetV1 dan ArcFace head/loss.
- `training/`: dataset loader, konfigurasi unfreeze, optimizer, checkpoint, dan training loop.
- `embeddings/`: ekstraksi dan penyimpanan embedding 512 dimensi.
- `verification/`: cosine similarity, threshold, pasangan genuine/impostor, dan metrik.
- `artifacts/`: checkpoint, model export, grafik, serta laporan hasil pelatihan.

## Catatan penting

Notebook Colab v2 dipakai sebagai catatan pelatihan saat ini. Untuk saat ini, jangan memindahkan training ke lokal. Kelak, bagian kode notebook dapat dipisah secara bertahap menjadi modul Python di `src/faceid/`.
