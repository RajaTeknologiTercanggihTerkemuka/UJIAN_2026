# Optional active liveness challenge protocol (not a PAD replacement)

A future browser flow may issue a random, short-lived challenge (for example, turn head left/right or blink), record a short sequence, and evaluate whether the requested action occurred in the expected order. Use one-time server-side nonces and short expiry.

**Limitations:** a blink/head-turn prompt is not a robust presentation-attack detector by itself. Replayed video, deepfake streams, virtual cameras, and real-time injection can bypass it. Do not return `live=true` solely because frontend code says an action was completed. Combine with a validated PAD model and conduct attack testing.
