# PAD validation protocol

## Dataset collection

Collect short camera sessions using the same camera paths intended for deployment (laptop webcams and representative phones). For each consenting participant, capture bona fide sessions under normal indoor lighting and expected pose variation. Separately collect controlled presentation-attack attempts using printed photographs and replayed images/videos on screens. Include 3D masks only if in the threat model and ethically/legally possible.

Do not use only downloaded images or still photos as PAD evaluation input: the upstream Silent-Face README explicitly warns that the intended evaluation uses camera-captured images and that device/scene changes can affect robustness.

## Split design

Split by participant and capture session/device where feasible. Avoid frames from the same video/session in both validation and test. Use validation data to choose `PAD_LIVE_THRESHOLD`; freeze it before evaluating the held-out test set.

## Metrics

Labels used in the supplied scripts:
- `1`: bona fide / real participant
- `0`: presentation attack

At a threshold:
- APCER = attack samples accepted as bona fide / all attack samples
- BPCER = bona fide samples rejected as attack / all bona fide samples
- ACER = (APCER + BPCER) / 2

Report APCER by attack instrument/type as well as aggregate APCER. State the camera/device, lighting, distance, resolution, model file/hash, crop scale, and threshold. If the sample size is small, include counts and confidence intervals; a single percentage can be misleading.

## Operational policy

- PAD `live`: continue to face similarity decision, provided PAD threshold was validated.
- PAD `spoof_suspected`: reject or send to operator review according to the study's decision policy.
- PAD `unknown`, `not_configured`, or error: `REVIEW_REQUIRED`, never assume live.

Blink/turn challenges may be an added active signal but are not a substitute for PAD. A 2D RGB PAD model may still be bypassed by untested attacks (e.g. virtual camera injection or sophisticated replay). Keep manual review and an alternative route available.
