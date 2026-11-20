# Face Recognition — model and inference module

This folder contains reusable model inference code. The training notebook V1 remains in `notebooks/`; the exported model is not bundled and must be copied from the user's Colab/Drive output.

## Model path

`artifacts/exported_models/imsfd_partial_ft_v1/inceptionresnetv1_imsfd_partial_finetuned.pth`

The loader supports the export dictionary containing `model_state` (as produced by notebook v2) or a plain state dict. It loads with `pretrained=None` so it does not silently download/reinitialize different weights.

## Preprocessing contract

- Camera: PIL RGB -> MTCNN detection -> exactly one face crop, size 160x160 -> fixed standardization `(pixel - 127.5) / 128` -> InceptionResnetV1.
- Training V1: SquarePad -> Resize 160x160 -> fixed standardization. This geometric preprocessing differs from an MTCNN crop. The inference threshold must be calibrated using validation images processed through the exact camera inference pipeline. If performance is poor, consider a controlled continuation fine-tuning with MTCNN-consistent crops, logged as a new model version.
- Quality checks return soft warnings. They should not automatically reject mild blur/brightness variation; only no face/multiple faces and severe decode failures are hard failures in this prototype.


## V1 compatibility note

V1 training preprocessing in the Colab notebook was `SquarePad -> Resize(160) -> fixed standardization`, not MTCNN. Camera inference uses MTCNN crop. Do not assume the V1 validation threshold transfers to MTCNN. Calibrate with a validation-only set that goes through this inference pipeline. If performance is not acceptable, a controlled continuation fine-tune with MTCNN-like crops may be justified; log it as a new version, do not overwrite V1.
