# FastAPI backend prototype

- `GET /api/health`: API/model readiness.
- `POST /api/registration/enroll`: admin-only reference-photo enrollment. A static reference photo does not prove liveness.
- `POST /api/verification/verify`: participant 1:1 verification; request includes NIM + captured image. Participant response contains only status, masked NIM, message, request id.
- `GET /api/admin/events`: admin-only operational diagnostics (detection confidence, face count/area, similarity, threshold, quality warnings, PAD score/status, timing breakdown, model/preprocessing/threshold versions).
- `/metrics`: Prometheus request counter and latency histogram; protect it in any shared deployment.

## Verification flow

Browser camera (`getUserMedia`) captures a still image -> FastAPI -> MTCNN requires exactly one face -> InceptionResnetV1 outputs a 512-d normalized embedding -> compare against the claimed participant's matching model/preprocessing template -> PAD must be bona fide when `REQUIRE_ANTISPOOF=true` -> return MATCH/MISMATCH/REVIEW_REQUIRED.

The API stores operational event metadata but does not persist raw uploaded frames. The face quality flags are advisory only. No-face/multi-face and uncertain PAD conditions trigger retry/review, not silent acceptance.

## Important limitations

This is a thesis prototype. Before any real UTBK use, add participant authentication/session binding, operator authentication beyond a prototype API key, rate limiting, HTTPS, security review, consent/retention controls, audited model/PAD thresholding, anti-presentation-attack evaluation, load testing, and a human review/alternative pathway.
