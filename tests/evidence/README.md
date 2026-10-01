# Historical hardware test evidence

These JSON records preserve measured timings, model/cache/context settings,
image IDs, GPU UUIDs, arithmetic correctness and deployment/restoration results.
Long generated text and token-ID arrays are omitted; source runner logs and full
outputs remain on N04 and in the local workspace referenced by the reports.

- `n04-2026-10-01/m10-result.json`: candidate four-M10 Q8/Q4 plus ordinary GGUF.
- `m10-q8-bonsai.json` / `m10-q4_0-bonsai.json`: Bonsai PQ2_0 candidate tests.
- `m10-q8_0-ordinary.json` / `m10-q4_0-ordinary.json`: separate Qwen4b regression.
- `m10-production-{deployment,validation}.json`: successful Q8 production rollout.
- `m60-production-q4-{test,validation}.json`: successful Q4 test, original service restored.
- `stock-deployment-result.json` / `stock-qwen-benchmark.jsonl`: separate Stock0.35.0 verification.

Bonsai runtime records qualify the exact local **Ollama0.34.1** Maxwell image.
They do not qualify the newer0.35.0 build, K80, every gamer card, or fully filled
190k prompts. M60 Q8 metadata support is not an actual Q8 inference test.
All Bonsai runs used published PQ2_0 weights; Q8/Q4 names describe KV cache.
The Q4 test is not a comprehensive model-quality evaluation.

These are historical artifacts, not fabricated unit-test fixtures and not input
for automatic release promotion. CI requires new evidence from the exact
immutable registry image and configured host for each release.
