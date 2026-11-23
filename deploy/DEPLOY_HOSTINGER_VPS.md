# Panduan Deploy ke VPS Hostinger (Aman Berdampingan dengan FastAPI Lain di Port 8000)

Paket ini menjalankan **FastAPI + PostgreSQL (`pgvector` HNSW)** di Docker pada port lokal **`127.0.0.1:8010`** sehingga **TIDAK bentrok** dengan aplikasi temanmu di port `8000` maupun PostgreSQL lain di port `5432`.

---

## Ringkasan Ukuran yang Masuk VPS
- Kode dari GitHub: **< 2 MB**
- 2 file bobot model `.pth` (via `scp`): **~96 MB**
- 10 akun demo (`deploy/demo_seed.sql`, via `scp`): **~80 KB**
- Dataset training 7 GB & checkpoint 393 MB **TIDAK ikut naik ke VPS**.

---

## Langkah 1 — Push Kode ke GitHub (dari Laptop)
Pastikan file berat sudah terabaikan (`.gitignore` sudah diatur otomatis):
```powershell
git status
git add .
git commit -m "Siapkan paket Docker VPS, pgvector HNSW, dan GitHub Actions CI/CD"
git push origin main
```

---

## Langkah 2 — Siapkan Folder & Clone Repo di VPS Hostinger (via SSH)
Masuk ke VPS Hostinger temanmu:
```bash
ssh root@IP_VPS_HOSTINGER
```

*(Jika RAM VPS ≤ 2 GB, buat Swap 2 GB sekali saja agar PyTorch CPU tidak kehabisan RAM:)*
```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Clone repo GitHub kamu & clone source kode arsitektur Silent-Face (kecil, hanya kode Python):
```bash
git clone https://github.com/USERNAME/REPO-KAMU.git ~/ta-v2-face
cd ~/ta-v2-face
git clone --depth 1 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git "Face Recognition/third_party/Silent-Face-Anti-Spoofing"
mkdir -p "Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1" "Face Recognition/artifacts/antispoof/weights"
```

---

## Langkah 3 — Upload 3 File Privat Sekali Saja (dari PowerShell Laptop via `scp`)
```powershell
# 1) Model InceptionResNetV1 Partial Fine-Tuning (~94 MB)
scp "Face Recognition\artifacts\exported_models\imsfd_partial_ft_v1\inceptionresnetv1_imsfd_partial_finetuned.pth" root@IP_VPS_HOSTINGER:"~/ta-v2-face/Face Recognition/artifacts/exported_models/imsfd_partial_ft_v1/"

# 2) Model Anti-Spoofing MiniFASNetV2 (~1.85 MB)
scp "Face Recognition\artifacts\antispoof\weights\2.7_80x80_MiniFASNetV2.pth" root@IP_VPS_HOSTINGER:"~/ta-v2-face/Face Recognition/artifacts/antispoof/weights/"

# 3) 10 Akun Demo Fauzi..Zahra (~80 KB)
scp "deploy\demo_seed.sql" root@IP_VPS_HOSTINGER:"~/ta-v2-face/deploy/"
```

---

## Langkah 4 — Buat `.env` & Jalankan Docker di VPS
Di SSH VPS:
```bash
cd ~/ta-v2-face
cp .env.vps.example .env
nano .env
```
Sesuaikan 3 baris di `.env`:
- `POSTGRES_PASSWORD=` (isi password acak yang kuat)
- `CORS_ORIGINS=https://ujian.domain-temanmu.com`
- `ADMIN_API_KEY=1234` *(bisa diganti lebih kuat setelah selesai demo)*

Lalu bangun & jalankan Docker:
```bash
docker compose up -d --build
```
Tunggu ~20 detik, lalu cek apakah sehat dan **`pgvector_hnsw`** aktif:
```bash
curl http://127.0.0.1:8010/api/health
```
Masukkan 10 akun demo (`04221001` Fauzi – `04221010` Zahra) ke PostgreSQL Docker, lalu reload vector index:
```bash
docker compose exec -T db psql -U face_app_user -d face_registration < deploy/demo_seed.sql
curl -X POST http://127.0.0.1:8010/api/admin/index/reload -H "X-Admin-Key: 1234"
```

---

## Langkah 5 — Sambungkan Subdomain + HTTPS (WAJIB untuk Kamera Webcam!)
1. Di panel DNS Hostinger temanmu, buat **A Record** baru:
   - Host: `ujian` (menjadi `ujian.domain-temanmu.com`) → Target: `IP_VPS_HOSTINGER`
2. Di VPS (minta temanmu bantu jika Nginx dikelola dia), salin `deploy/nginx-site.conf` ke Nginx:
   ```bash
   sudo cp ~/ta-v2-face/deploy/nginx-site.conf /etc/nginx/sites-available/ta-v2-face
   sudo nano /etc/nginx/sites-available/ta-v2-face   # ganti ujian.domain-kamu.com
   sudo ln -s /etc/nginx/sites-available/ta-v2-face /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   sudo certbot --nginx -d ujian.domain-temanmu.com
   ```
3. Buka **`https://ujian.domain-temanmu.com`** → Website + Kamera + Anti-Joki $1:N$ (`pgvector_hnsw`) langsung online tanpa mengganggu port `8000` temanmu!

---

## (Opsional) Aktifkan Auto-Deploy CI/CD GitHub Actions
Di repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**, tambahkan:
- `VPS_HOST` = IP VPS Hostinger
- `VPS_USER` = `root` (atau user sudo)
- `VPS_SSH_KEY` = isi private SSH key untuk login ke VPS
- `VPS_APP_DIR` = `~/ta-v2-face`

Setelah itu, setiap kamu `git push origin main`, GitHub Actions (`.github/workflows/deploy.yml`) otomatis mengecek kode lalu melakukan `git pull` + `docker compose up -d --build` di VPS!
