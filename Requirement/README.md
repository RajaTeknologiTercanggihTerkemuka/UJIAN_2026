# Requirement / Dependencies

Folder ini menyimpan **daftar dependensi**, bukan tempat menyimpan file library hasil download. Saat `pip install -r ...` dijalankan, paket dipasang ke Python environment yang aktif (`.venv` lokal atau runtime Colab).

File:

- `requirements-colab-training.txt`: paket tambahan untuk notebook training di Colab. Periksa `torch`/`torchvision` yang sudah tersedia di runtime sebelum memasang atau menggantinya.
- `requirements-local-inference.txt`: paket untuk nanti menjalankan inference lokal. Instalasi PyTorch/Torchvision harus dipilih sesuai OS dan CPU/CUDA dari halaman resmi PyTorch; jangan menebak versi.
- `requirements-api.txt`: paket backend web, dipakai pada tahap Website nanti.
- `requirements-dev.txt`: alat pengembangan/notebook.
- `requirements-legacy.txt`: daftar lama Anda, disimpan agar tidak hilang; bukan file instalasi utama.

Catatan: screenshot VS Code memperlihatkan interpreter Python 3.14. PyTorch saat ini mendokumentasikan dukungan Windows Python 3.10–3.14, tetapi itu **tidak otomatis menjamin kompatibilitas seluruh kombinasi** dengan `facenet-pytorch` dan dependensi lain. Rilis PyPI `facenet-pytorch` yang terlihat saat ini adalah 2.6.0 (29 April 2024). Karena training sudah dilakukan di Colab, jangan menginstal seluruh paket ke Python lokal sekarang. Kita validasi kebutuhan dan kombinasi versi saat mulai mengembangkan inference lokal.

Paket dari daftar lama yang bukan bagian notebook training saat ini: FastAPI, Uvicorn, Jinja2, dan python-multipart (untuk backend website nanti); `einops` tidak digunakan di notebook v2; OpenCV hanya diperlukan bila modul preprocessing/deteksi yang kita tulis menggunakannya; IPython terutama untuk interaktif/notebook.
