# Fork vs. stock Ollama

As of: 2026-10-08. Basis: Ollama v0.40.0, llama.cpp b11351 (pulled in via FetchContent; the patches also apply to v0.35.1 / b11232). The fork is **not a source fork in the
Git sense**: the Docker build fetches the official Ollama tag and applies Python patch scripts from `scripts/` to the Go and
llama.cpp sources. Every script is idempotent and aborts the build (fail closed) if the expected anchor is not found exactly
once in the upstream code.

## What the fork does differently (summary)

| Topic | Stock Ollama | Fork |
| --- | --- | --- |
| GPU targets | v0.35.1 and v0.40.0 CUDA 12 build (`llama/server/CMakePresets.json`): `50-virtual;52-virtual;60;61;70;75;80;86;89;90;90a;100;120`, i.e. Maxwell only as PTX (JIT on first start); CUDA 13 build: CC 75 and newer | native CUBINs (`-real`) for CC 5.0–9.0 in one image (no PTX JIT delay), CUDA 12.0.1 base; CUDA 11 image for K80 |
| Flash Attention | global on/off after detection | per tier: only if all participating GPUs support it |
| GPU selection | Ollama scheduler | dynamic pool (fast GPUs first, legacy GPUs only when needed) |
| Batch size | computed internally (measured `-b 2048`), `OLLAMA_MAX_BATCH_SIZE` has no effect | `OLLAMA_MAX_BATCH_SIZE` is honored |
| Context | as configured | as configured (the context halving of the pool patch applies only when Flash Attention is explicitly disabled) |
| Unknown GPU / CC | no comparison with the image's build targets | comparison with `CUDA_ARCHS`; default `mask`, `fail`/`ignore` selectable |
| MTP draft (qwen35moe) | on via model manifest (`draft_num_predict 2`), crashes on Maxwell (cuBLAS race) | `OLLAMA_DRAFT_NUM_PREDICT` is the default and an upper bound, also for manifest values; default 0 |
| Embedding table (`token_embd`) | always in host RAM | follows layer 0 into VRAM |
| Fit (nextn slot) | counts the slot only with MTP loaded; with MTP off layer 0 stays on the CPU (`41/42`) | always counts it (`42/42`) |
| Pipeline parallelism (multi-GPU) | automatic, fourfold input buffers in host RAM | can be switched off with `LLAMA_PIPELINE_PARALLEL=0` |
| CUDA graphs | disabled on legacy GPUs | can be enabled with `GGML_CUDA_GRAPHS_LEGACY=1` |
| More than 8 GPUs in one process | NCCL build forces VMM peer access on all devices; with 12 GPUs `peer mapping resources exhausted` (9 still run) | forcing only up to 8 devices (`patch-llama-vmm-peer-access.py`); 12× Tesla M60 run in one instance |
| Architecture `kolibri1` (Aleph Alpha Kolibri-1, 78B MoE) | `unknown model architecture: 'kolibri1'` | architecture added via `patches/kolibri1/` (`patch-llama-kolibri1.py`) |
| Jinja `tojson` | template error with some models | compatible |

## Patch scripts

Go (Ollama):

- `patch-ollama-fa.py` – tier-aware Flash Attention decision.
- `patch-ollama-dynamic-pool.py` – `selectGPUPool`: the model runs on the
  fast pool if it fits into its VRAM (threshold 75 %), otherwise on all GPUs.
  The halving of `-c` and `-np` when `OLLAMA_MAX_BATCH_SIZE` is set applies
  only if Flash Attention is explicitly off.
- `patch-ollama-batch.py` – `OLLAMA_MAX_BATCH_SIZE`.
- `patch-ollama-discovery.py` – compute capability by CUDA index, fail closed;
  `OLLAMA_ALLOW_UNKNOWN_CC` explicitly allows unknown cards.
- `patch-ollama-mtp-default.py` – `OLLAMA_DRAFT_NUM_PREDICT`: a value from the
  request wins; a value from the model manifest is capped to the variable
  (0 = MTP off); with neither, the variable is the default. Test:
  `TestDraftNumPredictServerDefault`.

llama.cpp:

- `patch-llama-tier-fitting.py` – fitting across mixed GPU tiers, split buffer removed.
- `patch-llama-fit-nextn.py` – always counts the nextn slot in the fit.
- `patch-llama-input-gpu.py` – `llama_model::load_tensors` places the
  input layer (token embedding) on the device of layer 0.
  `LLAMA_INPUT_LAYER_GPU=0` restores the upstream placement.
  Placement still goes through `select_weight_buft` (with CPU fallback).
  `LLAMA_ARG_OVERRIDE_TENSOR=token_embd.weight=CUDA0` bypasses this check and
  aborts the scheduler (N11-M10, b11232).
- `patch-llama-pipeline-parallel.py` – `LLAMA_PIPELINE_PARALLEL=0` skips
  pipeline parallelism; unchanged without the variable.
- `patch-llama-vmm-peer-access.py` – in NCCL builds `ggml_cuda_pool_vmm::alloc` forces peer access only up to 8 devices
  (CUDA allows 8 peers per mapping); with more than 8 GPUs only the owning device gets access.
- `patch-llama-kolibri1.py` – applies `patches/kolibri1/kolibri1-llama.cpp.patch` (runtime part of the patch from
  `Hob-forge/Kolibri-1-GGUF`, MIT): gating mode `SIGMOID_LOGIT_ADD`, model graph, tokenizer type `kolibri1`. Without the patch
  no Kolibri-1 GGUF loads. The script touches `src/CMakeLists.txt` so that the `models/*.cpp` glob is re-evaluated.
- `patch-llama-jinja-tojson.py` – `tojson` compatibility.
- `patch-llama-cuda-graphs-legacy.py` – opt-in for CUDA graphs on CC < 7.0.

## Which patches each image gets

| Patch | cuda11-legacy | cuda12-maxwell | cuda13-rtx |
| --- | --- | --- | --- |
| tier-fitting, fit-nextn, input-gpu, pipeline-parallel, vmm-peer-access, kolibri1, jinja-tojson | yes | yes | yes |
| fa, dynamic-pool, batch | yes | yes | yes |
| discovery, mtp-default | no | yes | yes |
| cuda-graphs-legacy | no | yes | no |

## Runtime scripts (in the image)

- `gpu-detect.sh` – detects GPUs, forms tiers and exports, among others,
  `OLLAMA_GPU_TIER_THRESHOLD`, `OLLAMA_FAST_GPU_DEVICES`,
  `OLLAMA_FAST_POOL_VRAM_GB`, `OLLAMA_DRAFT_NUM_PREDICT=0`. Checks cards against the image's
  `CUDA_ARCHS` (`OLLAMA_UNSUPPORTED_GPU=mask|fail|ignore`, default `mask`).
- `ollama-entrypoint.sh`, `auto-optimize.py`, `ollama-proxy.py`,
  `inject-presets.py` – startup, tuning, proxy and presets
  (`presets/gpu-targets.json`).

## Measurement tools (in the repo)

- `scripts/bench-throughput.sh` – decode and prefill with the **effective** values from the
  runner log (`n_ctx`, `n_batch`, `n_ubatch`, Flash Attention, KV type, layers).
- `scripts/sweep-batch.sh` – batch ladder per host (restarts the container).
- `scripts/compare-batch-quality.py` – compares answers at different batch sizes.

## Environment variables (fork-specific)

| Variable | Effect | Default |
| --- | --- | --- |
| `OLLAMA_MAX_BATCH_SIZE` | batch size (`-b`/`-ub`) | stock value |
| `OLLAMA_DRAFT_NUM_PREDICT` | MTP draft length, default and upper bound (also for manifest values), 0 = off | 0 (via gpu-detect) |
| `OLLAMA_UNSUPPORTED_GPU` | `mask`, `fail`, `ignore` | `mask` |
| `OLLAMA_ALLOW_UNKNOWN_CC` | allow unknown CC | off |
| `GGML_CUDA_GRAPHS_LEGACY` | CUDA graphs on legacy GPUs | off |
| `LLAMA_PIPELINE_PARALLEL` | `0` = pipeline parallelism off | on |
| `LLAMA_INPUT_LAYER_GPU` | `0` = embedding on the CPU as upstream | on |
| `OLLAMA_FORCE_GPU_LAYERS` | `1` = greedy fill: fill GPUs in descending CUDA index order, only as many as needed; replaces the fit | off |
| `OLLAMA_LAYER_OVERHEAD_SCALE` | factor on the weights per layer in the greedy fill (reserve for compute buffers and KV), 1.0–5.0 | adaptive |
| `LLAMA_ARG_FIT_TARGET` | fit reserve per GPU in MiB (llama.cpp variable) | upstream (~2 GiB) |

`LLAMA_ARG_FIT_TARGET` is upstream; the fork recommends `256` because, with the embedding in VRAM,
an expert tensor (210.82 MiB `ffn_down_exps`) otherwise spills into host RAM.

## Unchanged

API, model format, Modelfile syntax, registry and CLI match stock.
KV cache configuration (`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`) is as in
stock. The Bonsai path (`BONSAI=ON`) is optional and off in the current default
(`BONSAI=OFF`).

Measurements for 8–12 GPUs, greedy fill, context ladder and soak test: [TUNING.md](TUNING.md).
