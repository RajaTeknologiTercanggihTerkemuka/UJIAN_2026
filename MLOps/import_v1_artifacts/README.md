# V1 artifacts to copy from Google Drive

Copy only these files/folders from `IMSFD_ArcFace_Output_v2`:

- `training_history.csv`
- `validation_threshold_search.csv`
- `verification_metrics_test.json`
- `03_checkpoints/best_checkpoint.pth`, `03_checkpoints/last_checkpoint.pth`
- `04_plots/*.png`
- `05_model_export/inceptionresnetv1_imsfd_partial_finetuned.pth`
- `05_model_export/model_metadata.json`

Do NOT copy `identity_mapping.csv`, `dataset_manifest.csv`, student photos, masterdata, `registry_faces.pt`, or any file containing NIM/name into MLflow imports.
