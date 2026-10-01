# Ollama 0.35.0 hardware and rollout evidence

Captured on 2026-10-01; see the [fleet report](../../../FLEET-ROLLOUT-0.35.0-2026-10-01.md).

- final-audit.json: all14 APIs, image/container IDs, cache settings, contexts,
  GPU reservations, shared model mounts and preservation checks.
- n02-rollout.json / n11-rollout.json / n04-m10-rollout.json /
  n04-m60-rollout.json: successful replacement, regression and rollback records.
- ollama-tesla-bonsai-bonsai.json: full M10 Q8 native qualification; final
  serialized replacement reuses this exact image/GPU/config proof and adds
  fresh capability, correctness and256-token checks.
- m60-qualification-result.json and m60-*.json: Q4/190k and Q8/131k Bonsai
  qualification and ordinary-model regressions; Q8/190k explicitly not qualified.
- backend-isolation.json: real per-process Prism/Spark library mappings and
  Spark production restoration, without mixing private GGML enums.

Long generated response/thinking fields are shortened to excerpts; timing,
checks and configuration evidence remain. Model weights and full environment
secrets are not included. Context capacity was allocated, not filled completely.

Final default change: q4-default-rollout.json contains all12 Tesla Q4 replacements,
actual-cache flags and preservation checks; q4-final-audit.json supersedes the
initial F16/Q8 audit; q4-backend-isolation.json verifies restored Spark with Q4.
