# Anti-spoofing / Presentation Attack Detection roadmap

The API currently has a provider interface and a fail-safe unconfigured provider. This is deliberate: there is no honest way to claim liveness from MTCNN confidence, face similarity, blur, or a blink prompt alone.

Candidate repository: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (repository license shown as Apache-2.0). Its README describes silent PAD for printed photos, screens, silicone masks and 3D faces; it also states camera/device and scene variation affects real-world behavior. Use the repository's own input/patch pipeline and inspect its model weights/licensing. Do not use its reported demo metrics as a promise for your UTBK environment.

For the thesis, compare at least bona fide samples against print and screen-replay attacks; if possible add replay videos and device variations. Split train/validation/test by subject/session/device as appropriate to avoid leakage. Evaluate APCER, BPCER, ACER and ROC/DET curves under an explicit PAD protocol. Face-verification FAR/FRR are separate metrics from PAD errors.

Keep quality checks mild: no face / more than one face are capture conditions; brightness and blur are warnings first. A participant should be able to retry or request manual review rather than be silently rejected by a brittle heuristic.
