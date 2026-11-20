# TA v2 — Website Registrasi dan Verifikasi Wajah Berbasis MLOps

Workspace utama tugas akhir. Kode sumber dipisahkan dari data, hasil eksperimen/pelatihan, dan dependensi.

## Struktur utama

- `Website/`: aplikasi web; Front-end dan Back-end disiapkan untuk tahap berikutnya.
- `Face Recognition/`: kode deteksi, preprocessing, InceptionResnetV1, ArcFace, training, embedding, dan verifikasi.
- `Dataset/`: data asli, data hasil pengolahan, manifest, serta metadata dataset.
- `MLOps/`: disiapkan untuk tahap berikutnya.
- `Requirement/`: daftar paket Python yang dibutuhkan untuk tiap lingkungan.

## Lingkungan kerja

- Gunakan Google Colab untuk training GPU seperti yang sudah dilakukan.
- Gunakan VS Code untuk mengelola kode, konfigurasi, dokumentasi, audit, dan pengembangan aplikasi.
- Jangan menyalin seluruh dataset wajah atau file bobot besar ke GitHub.
- Buat virtual environment lokal `.venv` di root proyek (abaikan dari Git).

## Catatan data

Dataset IMSFD asli tetap dipertahankan dalam struktur sumbernya. Jangan memindahkan atau mengganti nama folder A1/A2/B1, Training/Testing, atau folder NIM dari sumber asli. Simpan hasil audit dan manifest sebagai metadata terpisah.

Foto wajah, NIM, label identitas, embedding, checkpoint, dan model terlatih harus diperlakukan sebagai data terbatas. Jangan unggah ke repositori publik.
