# Ollama v0.40.0 on N04-RTX and N02-M60, 2026-10-07

- `n02-ab-deployed-vs-v040.jsonl`: A/B on N02-M60 GPU 0–3 (`ab.py`): deployed image (`ollama-gaps:kolibri-20261006`, Ollama 0.35.1) against the CI candidate for v0.40.0, deployed configuration, two rounds each.
- `n04-replicas-v040-candidate-untuned-env.jsonl` and `n04-replicas-previous-image-same-env.jsonl`: replicas of the former `:11436` (4× M10) and `:11442` (2× M60) configuration on other ports (`n04-suite.py`), v0.40.0 candidate against the previous image. The environment is the untuned production environment of that time (no batch, `FIT_TARGET` or `DRAFT_NUM_PREDICT` setting). In the v0.40.0 file the Bonsai row is the expected failure (HTTP 500); in the other file the `qwen3.6:35b` row failed with HTTP 500.
- `n04-deployed-v040-tuned.jsonl`: the deployed v0.40.0 instances `:11436` and `:11442` (`post-deploy.py`).

All speeds are single runs. N04-RTX has 4 vCPUs and a high base load.
