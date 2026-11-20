# DVC data versioning

Start from the project root. Raw and processed face data are sensitive biometric data. Keep the DVC remote private and encrypted; DVC itself is not encryption. Do not push image data to a public remote or GitHub.

Example (local encrypted/external drive remote, not synced publicly):

```powershell
# from TA v2 root
 dvc init
 dvc add "Dataset/raw/IMSFD_original"
 dvc add "Dataset/raw/UTBK_reference_images"
 dvc add "Dataset/processed/IMSFD_mtcnn_v1"
 dvc remote add -d private-store "E:/TA-v2-private-dvc-cache"
 dvc push
 git add .dvc .gitignore dvc.yaml
 git commit -m "Add private dataset version pointers"
```

Adjust paths to your actual folder. Do not place the raw masterdata CSV/NIM mapping in public Git. If using cloud storage, select an access-controlled private bucket. DVC's Google Drive remote may require extra auth setup and should be tested with a small non-sensitive fixture first.

The first snapshot is post-V1: it helps future reproducibility but does not recreate the missing V1-time data lineage. Record source, collection date/version, preprocessing code commit, split manifest digest, and DVC hash for each future run.
