# N04 M60 production Q4 KV test

User authorized Q4 KV testing on the M60 production instance on2026-10-01.
The temporary test on `ollama-m60-guard:11442` COMPLETED successfully; the
original production container/image/configuration were then restored and are
healthy. No permanent Q4 rollout was requested or performed.

Hardware: original two Tesla M60 GPUs, compute capability5.2:

- `GPU-d34095b6-ff1d-6abb-9605-94a1b120f7b3`
- `GPU-2dc77b8a-fb51-6cfa-fd0c-2bb951f55b98`

Test image: hardware-qualified local Maxwell image
`sha256:c5ae3f17cfd9bc116dacc141bb5562bc92732aaf190e0ab3d3968a60f5aa24ab`,
Ollama0.34.1 / pinned Prismadfffbe, CUDA12, compiledsm50/sm52, FAON.
The original service was idle with no model loaded before the test. Mount,
GPU requests, port11442, restart policy and Docker HostConfig were preserved.
Existing `bonsai2:27b-pq2_0` used the shared `/opt/ollama/models:/root/.ollama`
pool; no GGUF downloads or duplicates.

Settings: requested context190000, actual runner allocation190208;
Q4_0 K and V, Flash Attention enabled, parallel1, batch128, seed42,
temperature0, native context-aware fitting. CUDA capability probe passed
f16/q8_0/q4_0 on both M60s; only Q4 was tested with actual M60 inference.

| Check | Result |
| --- | --- |
| Arithmetic19×23 and1000−237 | Correct437 and763 |
| Repeated128-token numbers decode | 7.516512 /7.507669tokens/s |
| 2048-token sustained decode | 7.235860tokens/s; completed2048 |
| GPU model layer placement | 65/65 |
| Total KV cache | 3343.50MiB (3.265GiB) |
| KV per GPU | 1671.75MiB |
| Observed total GPU memory | 6356 /6486MiB including model/compute/cache |
| CUDA errors / test-container restarts | None /0 |

This proves cache allocation and short-prompt inference/stability, not a fully
occupied190k prompt, comprehensive Q4 quality equivalence, or GPU execution
of every individual graph operation. No M60 F16 speed baseline was measured.
M10 Q4 results on the same image were approximately1.83tokens/s short and
1.79tokens/s sustained; the M60 difference does not isolate any one cause.

The first attempt failed before loading Bonsai because the replacement image
selected internal port11435 while Docker forwarded11434. Automatic rollback
restored the original container. The adapter now explicitly sets
`OLLAMA_HOST=0.0.0.0:11434` and `OLLAMA_INTERNAL_PORT=11434`; all33 local
regression checks passed and the complete real M60 retry succeeded.

Final restored production:

- Container ID `3812a7e9a55e64770262bcb1a6bdbeb6b740abe086fe9b363317ef3bfa5c2fd7`.
- Original image `sha256:2ac2bd3ad46984fe6c3f022b6bf8d2dd6d723f902414a3d7e55b1975ca9cd5c1`.
- Original F16 cache /FAOFF /default context32768, original restart policy.
- Running, healthy, no loaded models, matching the pre-test model state.
- Stock11434/11435 and M10Q8 production11436 retained exact container
  identities, start times and restart counts throughout the test.

Raw remote artifacts: N04 `/tmp/bonsai-m60-prod-q4-20261001/`:
`test-result.json` and `production-validation.json` both passed;
`test-result.json` also confirms restored=true. Logs and original inspect
remain available. Local raw copies:
`.bonsai-work/rollout-20260930/m60-production-q4-{test,validation}.json`.

M60 Q8 actual inference and a separate ordinary-model regression remain
outstanding for the full CI hardware matrix. The broader local v0.35.0 build
has compiled successfully but is not yet hardware-qualified or deployed.
