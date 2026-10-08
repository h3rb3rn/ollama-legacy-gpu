# Tuning and measurements (as of 2026-10-08)

The measurements below were taken on Ollama 0.35.1 / llama.cpp b11232 unless a section says v0.40.0 (llama.cpp b11351); see
[Ollama v0.40.0](#ollama-v0400-llamacpp-b11351) for the comparison of both bases.

Model `qwen3.6:35b` (Q4_K_M), context 262144, KV cache q4_0, Flash Attention on, `LLAMA_ARG_FIT_TARGET=256`,
Ollama 0.35.1 with the patches of this repository. Measurements use `scripts/bench-throughput.sh`: decode with the prompt
"Write a long story about a robot." (120 tokens, `temperature 0`, `seed 1`), prefill with ~2500 words (unique per run),
two runs each. **The effective values from the runner log are authoritative** (`llama_context: n_ctx / n_batch / n_ubatch`),
not the env file and not `api/ps`: stock Ollama ignores the batch variables and computes its own, and `api/ps` reports the
configured instead of the actual context.

## Configuration per instance

| Setting | N04-RTX | N02-M60 | N11-M10 |
|---|---|---|---|
| Cards in the pool | 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB) | 4× Tesla M10 (8 GiB) |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 64 | 64 |
| Pipeline parallelism | on | off (`LLAMA_PIPELINE_PARALLEL=0`) | off (`LLAMA_PIPELINE_PARALLEL=0`) |
| Layers / effective context | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode tok/s | 42.3 / 41.6 | 15.0 / 16.0 | 9.3 / 9.3 |
| Prefill tok/s | 1140 / 1172 | 91.4 / 92.9 | 24.2 / 24.1 |
| Counted by Ollama as "not in VRAM" | 0 MiB (0.0 %) | 0 MiB (0.0 %) | 0 MiB (0.0 %) |
| Host buffer `CUDA_Host compute` | 1029 MiB | 32.8 MiB | 32.8 MiB |

Weights, KV cache and compute buffers live in VRAM on all three instances. Only pinned staging buffers stay in host RAM:
the host buffer above, a ~1 MiB output buffer and a ~25 MiB CPU compute buffer of the vision part.

## Batch size

The batch size only determines how many prompt tokens are processed per pass. It speeds up prefill; decode (one token per
step) is independent of it. Batch ladder at context 262144, per host with the model and GPU count of the time of measurement
(N04-RTX: four GPUs; model version not uniform across hosts):

| Host | Env batch | Effective `-b` | Decode tok/s | Prefill tok/s | Compute buffer per GPU | Host buffer |
|---|---|---|---|---|---|---|
| N04-RTX (4 GPUs) | 64 | 64 | 40.5 / 40.6 | 624 / 651 | ~650 MiB | – |
| | 512 | 512 | 40.4 / 40.4 (40.9 / 40.8) | 1021 / 1175 (1014 / 1171) | ~1.6 GiB | 1029 MiB |
| | 1024 | 1024 | 41.0 / 40.9 | 887 / 1009 | ~2.7 GiB | 2057 MiB |
| | 2048 | 2048 | 37.6 / 37.6 (only 3 of 4 GPUs used) | 549 / 553 | ~1.75 GiB (3 GPUs) | 1040 MiB |
| N11-M10 | 64 | 64 | 9.0 / 9.0 | 23.9 / 23.8 | ~551 MiB | 32.8 MiB |
| | 128 | 128 | 9.1 / 9.1 | 32.2 / 32.2 | ~590 MiB | 65.3 MiB |
| | 256 | 256 | 9.3 / 9.3 | 40.2 / 40.2 | ~666 MiB | 130.3 MiB |
| | 512 | 512 | 9.2 / 9.1 | 49.1 / 49.2 | ~821 MiB | 260.3 MiB |
| | 1024 / 2048 | **512 (capped)** | 9.3 / 9.3 | 49 | ~821 MiB | 260.3 MiB |
| N02-M60 | 64 | 64 | 14.2 / 14.8 | 87 | ~551 MiB | 32.8 MiB |
| | 128 | 128 | 14.2 / 13.9 | 108.5 / 110.9 | ~589 MiB | 65.3 MiB |
| | 256 | 256 | 14.2 / 14.9 | 139.0 / 140.5 | ~666 MiB | 130.3 MiB |
| | 512 | 512 | 13.5 / 12.6 | 150 / 155 | ~821 MiB | 260.3 MiB |
| | 1024 / 2048 | **512 (capped)** | 13.8 / 13.4 | 154 | ~821 MiB | 260.3 MiB |

- **N04-RTX:** 512 is the optimum. 1024 is slower; 2048 leaves one GPU empty and drops in decode.
- **N11-M10, N02-M60:** higher env values are capped to `-b 512` on the runner command line (cause not investigated).
  512 fits with 42/42 but costs 260 MiB of host buffer, which Ollama shows as ~1 % CPU; at 64 it is 33 MiB and 0 %.
  That is why the two 8 GiB instances run with 64. Prefill at 512 is about twice as fast as at 64.
- **Quality:** `scripts/compare-batch-quality.py` (fixed prompts, greedy, freshly loaded instance per run) on N04-RTX: at
  batch 64 and 512, 6 of 6 answers are identical, including a needle-in-a-haystack test with an 18,044-token prompt; two
  runs at 512 also agree. Only Turing/Ampere were checked, not Maxwell (different Flash Attention kernels); 6 prompts,
  up to 200 tokens. Raw data: `docs/evidence/batch-quality-2026-10-05/`.
- **When 512 pays off:** prompt lengths from the N04-RTX runner logs (756 requests): median 210 tokens, 75 % below 1143,
  90 % below 28,885, maximum 181,904; 40 % below 64 tokens (no difference), 21 % at least 4096, 7 % at least 32,768.
  The lines carry no timestamp and count only newly computed tokens (prompt cache hits are missing).

## Pipeline parallelism

With several GPUs in a layer split, llama.cpp enables pipeline parallelism when all layers are offloaded
(`n_gpu_layers > n_layer_all`) and no tensor overrides are active. The scheduler then keeps four copies of its input buffers
(`sched copies = 4`): the pinned host buffer and the compute buffers grow.

| Host | Pipeline | Host buffer | Compute buffer per GPU | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|
| N11-M10 (batch 64) | on | 129.6 MiB (`RssShmem` 146 MiB) | 648 MiB | 9.2 / 9.2 | 24.1 |
| | off | 32.8 MiB (`RssShmem` 47 MiB) | 551 MiB | 9.3 / 9.3 | 24.2 / 24.1 |
| N04-RTX (batch 512, 3 GPUs) | on | 1029 MiB | ~1.6 GiB | 42.3 / 41.6 | 1140 / 1172 |
| | off | 260.3 MiB | 822 MiB | 38.3–39.7 (mean 39.3) | 917–932 (mean 922) |

On Maxwell the pipeline brings no advantage and costs pinned RAM: it is off there. On N04-RTX it brings about +7 %
decode and +20 % prefill for 770 MiB more host buffer: it stays on there. `LLAMA_PIPELINE_PARALLEL` makes the choice.
With `OLLAMA_GPU_AUTODETECT=1` (N02-M60) the fork's fit produces tensor overrides (fit duration 22.7 s instead of 0.9 s), which
also switch the pipeline off.

## Host buffer

`CUDA_Host compute buffer` ≈ `n_ctx × n_ubatch × 2 bytes` (attention mask), times four with the pipeline:

| n_ctx | n_ubatch | Pipeline | computed | measured |
|---|---|---|---|---|
| 131,072 | 64 | off | 16 MiB | 18.9 MiB |
| 131,072 | 256 | off | 64 MiB | 68.3 MiB |
| 262,144 | 64 | off | 32 MiB | 32.8 MiB |
| 262,144 | 512 | off | 256 MiB | 260.3 MiB |
| 262,144 | 64 | on | 128 MiB | 129.6 MiB |
| 262,144 | 512 | on | 1024 MiB | 1029.1 MiB |

The rest of the process RSS (~0.8–1.0 GiB anonymous, ~0.22–0.28 GiB file mappings of the CUDA libraries) is runtime overhead,
not model data.

## "CPU" display in `ollama ps`

`/api/ps` returns `size` and `size_vram`; the percentage is `(size − size_vram) / size`. On N02-M60 this was 286 MiB (1.12 %)
at batch 512, 58 MiB (0.24 %) at batch 64 and 0 MiB with the pipeline switched off. On N04-RTX Ollama reports 0 MiB although
the runner allocates 1029 MiB of pinned host buffer: the display does not reflect the host buffer on every host.
`CUDA_Host compute buffer` in the runner log is reliable.

## N04-RTX: three versus four GPUs

Same image, context 262144, batch 512, pipeline on, same script:

| | 4 GPUs (2 runs) | 3 GPUs: 2× RTX 2060 + 1× RTX 3060 (4 runs) |
|---|---|---|
| Decode tok/s | 40.4 / 40.4, 40.9 / 40.8 (mean 40.6) | 42.7 / 42.2, 42.2 / 42.7, 41.7 / 42.4, 42.1 / 42.3 (mean 42.3) |
| Prefill, 2nd measurement | 1175 / 1171 | 1177 / 1174 / 1160 |
| Total VRAM | 31,198 MiB | 29,443 MiB |
| Free VRAM per GPU | 3.0 / 4.4 / 5.7 / 4.9 GiB | 1.3 / 3.4 / 2.7 GiB |

With three GPUs decode is about 4 % higher (one pipeline stage fewer; not verified). The context is reserved up front and
does not grow. The fourth RTX 3060 is free for other tasks.

## More than 9 GPUs in one instance (N02-M60, 12× Tesla M60)

The containers get only the cards under test (`--gpus device=<UUIDs>` and a matching `CUDA_VISIBLE_DEVICES`); Ollama 0.35.1,
context 262144, q4_0, batch 64, `LLAMA_PIPELINE_PARALLEL=0`. A first attempt that only shrank `CUDA_VISIBLE_DEVICES` but passed all
12 cards into the container was invalid (the runner still saw all 12) and is not used.

| Image | Model | GPUs in the container | Result |
|---|---|---|---|
| `pipefix-20261005` (no patch) | `qwen3.6:35b` | 8 | runs, 11.2–11.5 tok/s decode |
| `pipefix-20261005` (no patch) | `qwen3.6:35b` | 9 | runs, 10.7–10.8 tok/s decode |
| `pipefix-20261005` (no patch) | `qwen3.6:35b` | 12 | `CUDA error: peer mapping resources exhausted` on first load (`cuMemSetAccess` in `ggml_cuda_pool_vmm::alloc`, `ggml-cuda.cu:635`) |
| `vmmpeer-20261006` (with `patch-llama-vmm-peer-access.py`) | `qwen3.6:35b` | 12 | runs, 10.4 tok/s |
| `kolibri-20261006` (with patch) | Kolibri-1 | 8, 9, 10, 11, 12 | runs, values below |

Not measured: 10 and 11 GPUs without the patch (Kolibri-1 does not load in an image without the Kolibri patch; not repeated with Qwen).

**Cause:** `libggml-cuda.so` is linked against NCCL (`libnccl.so.2` in `ldd`), so `use_peer_access` in `ggml_cuda_pool_vmm::alloc`
is always true and `cuMemSetAccess` receives access descriptors for all visible devices. CUDA allows at most 8 peers per mapping, i.e.
one device plus 8 peers = 9 GPUs. The patch forces peer access only up to 8 devices; this is a conservative choice, since 9 GPUs also run
without the patch. Above that, only the owning device gets access.

`qwen3.6:35b` is slower on 12 GPUs at 10.4 tok/s than on the 4-GPU pool (13.9–14.8 tok/s in this measurement series, 15.0–16.0 tok/s in
steady operation of the pool): more stages and no gain for a model that fits on four cards.

## Kolibri-1 on N02-M60

`hf.co/Hob-forge/Kolibri-1-GGUF:Q4_K_M`: architecture `kolibri1`, 78.1B parameters (384 experts, 6 active), 47.5 GB, 51 layers, native context window
262144 (GGUF `kolibri1.context_length`), four of five layers with a sliding window of 513, every fifth with full attention without positional encoding.
Without `patch-llama-kolibri1.py` loading aborts with `unknown model architecture: 'kolibri1'`. Image `kolibri-20261006`, batch 64, q4_0,
`-c 262144`, all 51 layers in VRAM, nothing in RAM, load time 253 s.

**GPU count** (decode: 120 tokens, `temperature 0`; prefill: 2500 words; fit with spread, except the last row):

| GPUs | Total VRAM | Highest GPU | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|---|---|
| 8 | 51.8 GB | 7.5 GB | 11.1 / 10.9 | 68.9 |
| 9 | 52.4 GB | 6.7 GB | 10.8 / 10.7 | 67.5 |
| 10 | 53.1 GB | 6.7 GB | 10.7 / 10.5 | 66.6 |
| 11 | 53.2 GB | 5.8 GB | 10.4 / 10.4 | 65.0 |
| 12 | 53.3 GB | 5.8 GB | 10.5 / 10.4 | 65.0 |
| 12 visible, greedy fill 1.10 (uses 8) | 51.3 GB | 7.2 GB | 11.1 / 10.9 | 67.4 |

Each additional GPU costs about 0.1–0.2 tok/s decode and 1–2 tok/s prefill; no CUDA error occurred in any configuration.

**GPU reduction** with `OLLAMA_FORCE_GPU_LAYERS=1` (greedy fill, highest CUDA index first, only as many GPUs as needed), 12 visible:

| `OLLAMA_LAYER_OVERHEAD_SCALE` | GPUs under load | Total VRAM | Highest GPU | Result |
|---|---|---|---|---|
| 1.10 | 8 | 51.3 GB | 7.2 GB | runs, 11.1 / 10.9 tok/s; stable in the soak test (see below) |
| 1.03 | 7 | 50.9 GB | 8.1 GB | loads, then `CUDA error: out of memory` on the first request, reload (254 s per attempt) |

With 1.10 the reserve per card is about 0.9 GB; 7 GPUs leave no room for the compute buffers of a request. `OLLAMA_FORCE_GPU_LAYERS`
replaces the fit for every model of the instance (other models not measured with greedy fill).

**Context ladder** (12-GPU single instance with greedy fill on 8 GPUs, Ollama 0.35.1, two needles at 25 % and 80 % depth, `temperature 0`, `think true`):

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) | Duration | Needles found | VRAM peak per GPU |
|---|---|---|---|---|---|
| 9,421 | 58.5 | 11.8 | 3 min | 1/1 | – |
| 36,554 | 50.6 | 6.4 | 12.6 min | 2/2 | 7,314 MiB |
| 73,159 | 42.3 | 4.6 | 29 min | 2/2 | 7,314 MiB |
| 147,213 | 31.7 | 3.7 | 78 min | 2/2 | 7,314 MiB |
| 221,383 | 25.2 | 3.0 | 148 min | 2/2 | 7,314 MiB |
| 250,112 | 23.3 | 2.7 | 180 min | 2/2 | 7,314 MiB |

VRAM does not grow with the prompt length (KV cache and compute buffers are reserved up front at `-c 262144`); the temperature stayed ≤ 59 °C.
Decode falls from 11 to 2.7 tok/s, prefill from 69 to 23 tok/s. A first run with "245k" was invalid: the prompt had about 282k tokens, was truncated
to 131,074 tokens and the first needle was lost; the repeat with 250,112 tokens (about 95 % of the window) passes.
Raw data and driver script: `docs/evidence/kolibri-n02-2026-10-07/`.

**Soak test** (60 min, greedy fill on 8 GPUs, alternating reasoning arithmetic, tool call and 1500-word prompt, sequentially):
220 requests, 0 errors, decode min / median / max 10.7 / 12.1 / 17.3 tok/s, longest request 50.7 s, VRAM per GPU constant at 7,314 MiB (start = end),
temperature ≤ 55 °C, 0 CUDA errors in the runner log. Driver Xid messages could not be checked (`dmesg` and the kernel journal are not readable for
the test user).

Functional check (image `kolibri-20261006`): German fluent; reasoning arithmetic correct (80); tool call `get_weather` with `{"city": "Heidelberg"}`.
With `think: false` the reasoning ends up in `content` instead of `thinking`; with `think: true` it is separated (not investigated further).

## Ollama v0.40.0 (llama.cpp b11351)

**What changed upstream between 0.35.1 and 0.40.0** (26 commits, 206 files): MLX is the default on Apple Silicon (`mlxrunner`, decision
models on MLX), OpenAI-compatible tool-message fixes, multimodal embeddings, the llama.cpp bump b11232 → b11351 (119 upstream commits:
CUDA fixes for sm70/Volta, `llama_batch_ext` migration including speculative decoding, new models such as GLM-5.3-Flash; no Kolibri-1),
a changed `002-clef.patch` and a **manifest migration** (`manifests-v2`, `metadata`, package `compatmigrate`: Ollama-format models are
converted to llama.cpp-compatible manifest-list children in the background; the first load still uses the compat patch). The scheduler,
`envconfig`, GPU discovery and fit files are unchanged, so the Go patches of this repository apply as before. All patch scripts apply to
v0.40.0 (CI builds of all three variants succeed; the scripts fail closed).

**A/B on N02-M60, GPU 0–3** (4× Tesla M60, deployed configuration: batch 64, pipeline off, `FIT_TARGET` 256, MTP off, context 262144,
q4_0, `qwen3.6:35b`; the two images alternate, three decode runs each; `docs/evidence/n04-n02-v040-2026-10-07/n02-ab-deployed-vs-v040.jsonl`):

| Image | Round | Decode (tok/s) | Prefill (tok/s) | Load | Layers / batch / host buffer |
|---|---|---|---|---|---|
| deployed (Ollama 0.35.1, `kolibri-20261006`) | 1 | 13.2 / 13.9 / 15.0 | 84.4 | 142 s | 42/42, 64, 32.8 MiB |
| v0.40.0 (CI candidate `a513b46`) | 1 | 13.3 / 13.6 / 14.8 | 84.6 | 142 s | 42/42, 64, 32.8 MiB |
| deployed (0.35.1) | 2 | 13.1 / 14.9 / 15.8 | 84.9 | 141 s | 42/42, 64, 32.8 MiB |
| v0.40.0 | 2 | 13.3 / 15.0 / 15.3 | 85.2 | 141 s | 42/42, 64, 32.8 MiB |

Equal within the noise. `gpu-detect.sh`, the entrypoint, the proxy and `auto-optimize.py` are byte-identical in both images; the CI image
builds ten CUDA architectures (`libggml-cuda.so` 988 MB) instead of five (441 MB), which does not change the throughput.

**N04-RTX replicas with the untuned production environment** (the environment of the former `ollama-tesla-bonsai` / `ollama-m60-guard`:
no batch setting, no `FIT_TARGET`, `OLLAMA_DRAFT_NUM_PREDICT` unset; replicas on other ports, same GPUs; raw data in `docs/evidence/n04-n02-v040-2026-10-07/`):

| Replica | Model | Previous image (Ollama 0.34-era Bonsai build) | v0.40.0 candidate |
|---|---|---|---|
| 4× M10, context 190000 | `qwen3.5:9b` | 3.4 tok/s, 33/34 layers | 4.0–4.1 tok/s, 34/34 |
| 4× M10 | `qwen3.6:35b` | HTTP 500 (2 CUDA errors) | 2.2–2.7 tok/s, 42/42, MTP active |
| 4× M10 | `sovereign-judge-olmo31-32b` | 0.9–1.0 tok/s | 0.6–0.8 tok/s |
| 4× M10 | `bonsai2:27b-pq2_0` | 0.7–0.8 tok/s | HTTP 500 (expected: needs the Bonsai build) |
| 2× M60, context 32768 | `llama-guard3:8b` | 25.9 / 31.2 tok/s | 26.5 / 31.1 tok/s |
| 2× M60 | `qwen3.5:9b` | 12.9 / 11.9 tok/s | 12.8 / 12.6 tok/s |
| 2× M60 | `moe-sovereign-planner-9b` | 12.5 / 12.3 tok/s | 13.0 / 13.0 tok/s |

Single runs on a host with 4 vCPUs and high base load. The low M10 numbers are a configuration effect (batch 512/2048, MTP active) and the
host, not the Ollama version; with the tuned configuration below the same hardware reaches the N11-M10 level.

**N04-RTX deployed with v0.40.0 and the tuned configuration** (`ollama-m10` on `:11436`, `ollama-m60-guard` on `:11442`, image
`ollama-gaps:v040-20261007`, built locally with the architectures `50;52;61;75;86`):

| Instance | Model | Decode (tok/s) | Prefill (tok/s) | Load | Layers / context / batch | CUDA errors |
|---|---|---|---|---|---|---|
| `:11436`, 4× M10 | `qwen3.6:35b` | 9.0 / 8.9 / 8.7 | 31.6 | 123 s | 42/42, 262144, 64 | 0 |
| `:11442`, 2× M60 | `llama-guard3:8b` | 29.6 / 32.5 / 32.1 | 228 | 29 s | 33/33, 32768, 1024 | 0 |
| `:11442`, 2× M60 | `qwen3.5:9b` | 13.1 / 13.2 / 13.2 | 167 | 35 s | 34/34, 32768, 1024 | 0 |

For reference N11-M10 (also 4× M10, Ollama 0.35.1): 9.3 tok/s decode and 24.2 tok/s prefill.

**GitHub-built image on the four N04-RTX instances and on N11-M10** (2026-10-08; image from GitHub Actions run 37688616665, revision `738c694`, digest-pinned;
same models and script before and after, raw data in `docs/evidence/n04-github-image-rollout-2026-10-08/`):

| Instance | Model | Before (decode / prefill) | After (decode / prefill) | Layers / on GPU |
|---|---|---|---|---|
| `:11434`, RTX 2060 ×2 + RTX 3060 (fork 0.35.1 → v0.40.0) | `qwen3.6:35b` | 42.7 / 42.8 / 42.3 tok/s, 1149 tok/s | 42.8 / 40.1 / 41.2 tok/s, 1028 tok/s; repeats 41.9 / 37.8 / 40.7, 1148 tok/s and 39.5 / 40.1 / 41.0 | 42/42, 100 % |
| `:11435`, RTX 2060 + GTX 1060 (stock 0.35.0 → fork v0.40.0) | `moe-sovereign-planner-9b` | 22.7 / 22.9 / 21.9 tok/s, 585 tok/s | 24.3 / 24.4 / 24.4 tok/s, 697 tok/s | 33/33, 100 % |
| `:11436`, 4× M10 (local v0.40.0 build → CI image) | `qwen3.6:35b` | 9.0 / 8.9 / 8.7 tok/s, 31.6 tok/s | 8.9 / 8.9 / 8.8 tok/s, 32.5 tok/s | 42/42, 100 % |
| `:11442`, 2× M60 (local v0.40.0 build → CI image) | `llama-guard3:8b` | 29.6 / 32.5 / 32.1 tok/s, 228 tok/s | 29.8 / 31.5 / 31.1 tok/s, 227 tok/s | 33/33, 100 % |
| N11-M10 `:11434`, 4× M10 (fork 0.35.1 → v0.40.0) | `qwen3.6:35b` | 8.7 / 8.9 / 8.9 tok/s, 24.8 tok/s | 9.3 / 9.3 / 9.2 tok/s, 24.8 tok/s | 42/42, 100 % |

No CUDA errors. The CI image builds ten CUDA architectures instead of five; the Tesla instances show no difference. On `:11434` the decode is about
4 % lower on average (40.7 against 42.5 tok/s over the repeats) while the prefill is equal; with single runs on a host with 4 vCPUs this is within
the spread, a regression is neither shown nor excluded (an interleaved A/B on that instance was not run). `:11435` gains 7 % decode and 19 % prefill
from the fork build and v0.40.0. The v0.40.0 logs warn about models without a chat template (`moe-sovereign-planner-olmo3-7b`) and report the
Bonsai models as unreadable (`unsupported tensor "output.weight"`).

**Model store and the manifest migration.** v0.40.0 creates `manifests-v2` and `metadata` next to the legacy `manifests` directory and
converts Ollama-format models in the background (disk estimate: converted size + 25 % + 512 MiB). The N04 containers share
`/opt/ollama/models` with other Ollama versions, so the two v0.40.0 instances mount `blobs` and the legacy `manifests` read-only and write
to a private directory (`/opt/ollama/models/n04-v040`, `…/n04-v040-guard`). With a read-only store the models load and run through the
compat path (see the results above); the background conversion does not happen, and models cannot be pulled or created through these instances.
`/opt` is a separate 916 GB disk on N04 (285 GB free); the 23 GB free are the Docker root file system. Whether the conversion with a
writable store changes the layout in a way that older versions cannot read was not tested.

## Further findings

- **MTP (speculative decoding):** slower than without on N04-RTX (4 GPUs): 40.3 tok/s without MTP, 38.0 with `draft_num_predict 2`,
  28.4 with 4 (running-text prompt). It crashes on Maxwell. `OLLAMA_DRAFT_NUM_PREDICT=0` is the default.
- **CUDA graphs on Maxwell** (`GGML_CUDA_GRAPHS_LEGACY=1`): +35 % decode on N11-M10 (6.4 → 8.6 tok/s, CPU-weak i5-3470T,
  main thread at 100 %), no effect on N02-M60 (12.4–13.2 versus 12.6–13.7). Runs on N11-M10.
- **Thread count** (`LLAMA_ARG_THREADS` 4 / 2 / 1) has no effect with full GPU offload (N11-M10, 8.95 tok/s).
- **Stress test** (8025-token prompt, 1500-token decode, batch 512): N11-M10 prefill 45.7 / decode 7.08 tok/s, N02-M60 148.6 / 9.32
  tok/s; both without errors, VRAM unchanged afterwards.
- **Stock Ollama 0.35.0 on N04-RTX** (4 GPUs, context 262144, env batch 64): effective `-b 2048`, `41/42` layers, the third
  GPU practically unused; 31.1–31.3 tok/s decode, 277–281 tok/s prefill. The fork loads `42/42` on all GPUs.
- **Without `LLAMA_ARG_FIT_TARGET=256`** (default fit target) `ffn_down_exps` (210.82 MiB) spills into host memory.

## Limits

- Few runs per configuration, no analysis of variance; measurements on hosts with different background load (ComfyUI, further
  Ollama containers) can deviate.
- Prefill and decode at very long context (> 32k) for models other than Kolibri-1, other models and parallel requests are not measured.
- The cause of the remaining CPU binding of the main thread on Maxwell is not fully understood (no profiling).
