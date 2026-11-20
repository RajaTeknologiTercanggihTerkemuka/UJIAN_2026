# Presentation Attack Detection (PAD) / Anti-spoofing

`MTCNN` is the face detector/cropper. It is not a liveness detector. PAD is a separate model.

The adapter `silentface_provider.py` loads MiniFASNet weights from the official Silent-Face-Anti-Spoofing repository. It receives the **original RGB frame** and the MTCNN bounding box, then builds the model-specific expanded crop (e.g. scale 2.7) at 80x80. The official implementation uses three classes and index 1 is the real/bona fide class; see the upstream `test.py` and `generate_patches.py`.

## Setup

1. Clone the official repository under `Face Recognition/third_party/Silent-Face-Anti-Spoofing/`.
2. Download the official `2.7_80x80_MiniFASNetV2.pth` to `Face Recognition/artifacts/antispoof/weights/` (PowerShell helper is in `Face Recognition/scripts/download_silentface_weights.ps1`).
3. Set `ANTI_SPOOF_PROVIDER`, `PAD_REPO_ROOT`, and `PAD_WEIGHTS_DIR` in `.env`.
4. Leave `PAD_LIVE_THRESHOLD` blank initially. The provider returns `unknown`; with `REQUIRE_ANTISPOOF=true`, final verification remains `REVIEW_REQUIRED` until PAD threshold calibration is completed.
5. Capture a labelled PAD validation set with the target webcam/phone devices: bona fide captures and presentation attacks (at least printed-photo and replay-screen attacks; evaluate masks if in threat model). Separate by subject/session/device when possible.
6. Use the PAD evaluation scripts to select a threshold on validation data, then freeze it and evaluate on a distinct test set. Do not tune on the test set.

## Important limitations

- The upstream repo reports that performance depends on camera model and scene; it recommends camera-captured test images and a normally visible, reasonably upright face. Its README documents print/display/silicone/3D attacks as target attack types.
- A published pretrained model is a starting point, not proof of acceptable performance on this UTBK population or target cameras. Report APCER, BPCER, ACER and a thresholded confusion matrix on your own capture protocol.
- Never return `live` solely because a file exists or model inference ran. Missing threshold, failed PAD inference, or unknown output must fail closed to manual review.
- A blink/head-turn challenge can complement PAD but is not a substitute for a trained PAD model.
- Preserve upstream LICENSE/NOTICE and record the exact Git commit and SHA256 of the model weight. Check whether the model weights have any separate usage terms before non-academic/commercial deployment.
