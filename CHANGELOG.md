# Changelog

All notable changes to this fork are documented here.

---

## [Unreleased] — 2026-10-08

- **N04-RTX rollout:** all four Ollama instances (`:11434`, `:11435`, `:11436`, `:11442`) run the GitHub Actions image of run 37688616665 (Ollama v0.40.0, revision `738c694`, digest-pinned). Validation against the previous state with the same models: `:11435` (stock 0.35.0 → fork) +7 % decode, +19 % prefill; the Tesla instances unchanged; `:11434` prefill equal, decode about 4 % lower on average (single runs). See [docs/TUNING.md](docs/TUNING.md).
- **v0.40.1 on N04-RTX:** all four instances (`:11434`, `:11435`, `:11436`, `:11442`) switched to the GitHub Actions image of run 37744806101 (Ollama v0.40.1, revision `b22a75e`, `cuda12-maxwell-0.40.1`). Same models and script as for v0.40.0: decode and prefill equal within the noise (`:11434` 40.0–42.5 tok/s, `:11435` 23.6–24.1, `:11436` 8.8–8.9, `:11442` 30.3–32.8), 100 % on the GPUs, no CUDA errors, no restarts. Git tag `v0.40.1` added. N11-M10 and N02-M60 stay on v0.40.0.
- **N11-M10 rollout:** `ollama` (:11434, 4× Tesla M10) runs the same GitHub Actions image (v0.40.0, digest-pinned) from `docker-compose.m10-pool.yml`; `qwen3.6:35b` 9.3 / 9.3 / 9.2 tok/s decode and 24.8 tok/s prefill (before: 8.7–8.9 and 24.8), 42/42 layers, no CUDA errors. The previous container is kept stopped.
- **N02-M60 rollout:** the nine AI-Village endpoints (pool `:11434`, single GPUs `:11435`–`:11442`) run the same GitHub Actions image, swapped one endpoint at a time while the agents were active (about one minute per endpoint, the model of each agent reloaded with its context). All eight single-GPU endpoints load fully on the GPU (100 %, no CUDA errors, 14.2–23.6 tok/s one at a time); the King's pool generates at the level of the previous container under the same load (8.3 against a median of 7.8 tok/s). The previous containers are kept stopped.
- **Caveat from the N02-M60 rollout:** check requests sent to the King's pool while the agents were active caused repeated model reloads (7 loads in 65 minutes against 2 in 15 hours); the pool was stable once the checks stopped. Live village endpoints should only be observed (`/api/ps`, container log). See [docs/FLEET-STATE.md](docs/FLEET-STATE.md).
- **Versioning:** git tags `vX.Y.Z` mark the last commit based on Ollama `vX.Y.Z` (added for v0.30.10, v0.30.11, v0.32.15, v0.33.2, v0.33.3, v0.34.0, v0.34.1, v0.34.2, v0.34.4, v0.35.1, v0.40.0 and v0.40.1). Every non-Bonsai CI build additionally tags its image `<variant>-<ollama version>`; the first run with this rule (37744806101, revision `b22a75e`) produced `cuda12-maxwell-0.40.1`. See [docs/VERSIONING.md](docs/VERSIONING.md).
- **CI builds the newest Ollama release:** run 37744806101 built Ollama v0.40.1 (nine commits after v0.40.0: cloud usage API proxy, Windows fixes, MLX patch, documentation; no llama.cpp or CUDA change). The v0.40.1 image was later validated on N04-RTX (see above). There is no image tag `cuda12-maxwell-0.40.0` (the v0.40.0 build predates the version tags).
- **Documentation:** README status section and measured values updated to the GitHub-image measurements; runtime status of the GPU table extended (GTX 1060 and the M60 variants tested); the release-automation note about stock Ollama on N04 `:11434`/`:11435` removed.

## [Unreleased] — 2026-10-07

- **Based on:** Ollama v0.40.0 (llama.cpp b11351); `OLLAMA_VERSION` defaults in all three Dockerfiles are `v0.40.0` (was `main`). All patch scripts apply unchanged. A/B on N02-M60 (4× Tesla M60, deployed configuration, `qwen3.6:35b`): decode 13.1–15.8 tok/s on both bases, prefill 84–85 tok/s, load time 141–142 s. Details in [docs/TUNING.md](docs/TUNING.md).
- **Image:** `ollama-gaps:v040-20261007` (`cuda12-maxwell`, CUDA 12.0.1, archs 50;52;61;75;86, FA=ON, `BONSAI=OFF`); the CI builds the same Dockerfile with ten architectures and Ollama `latest`.
- **Deployed on N04-RTX:** `ollama-m10` (:11436, 4× Tesla M10, tuned like N11-M10: `qwen3.6:35b` 8.7–9.0 tok/s decode, 31.6 tok/s prefill, 42/42 layers, context 262144) and `ollama-m60-guard` (:11442, 2× Tesla M60) run on v0.40.0. They replace `ollama-tesla-bonsai` (Bonsai build) and the previous guard; Bonsai models are no longer served there. The shared model store is mounted read-only because v0.40.0 migrates the manifest layout (`manifests-v2`, `metadata`).
- **Upstream changes read for this update:** release notes 0.35.0 / 0.35.1 / 0.40.0, the 26 commits between v0.35.1 and v0.40.0 and the 119 llama.cpp commits b11232…b11351 (summary in docs/TUNING.md).
- **Deployed on N02-M60:** nine endpoints for the AI-Village agents on `ollama-gaps:v040-20261007` (pool on GPU0-3 `:11434` for 01-king, single-GPU instances `:11435`–`:11442` for agents 02-09), context per instance as requested by the agent (131072 or 262144). Every agent's model loads fully on the GPU at its context (batch 64, no CUDA errors; `qwen3.6:35b` 13.5 tok/s, the 4B models 6.5–14.6 tok/s). The 12-GPU Kolibri-1 instance is stopped; its layout stays available as `docker-compose.single12.yml`. Data: `docs/evidence/n02-village-2026-10-07/`.

## 2026-10-06

- **Based on:** Ollama v0.35.1 (llama.cpp b11232)
- **Image:** `ollama-gaps:kolibri-20261006` (`cuda12-maxwell`, CUDA 12.0.1, archs 50;52;61;75;86, FA=ON, `BONSAI=OFF`)

- **VMM peer access:** `patch-llama-vmm-peer-access.py` limits the forced peer access of the CUDA VMM pool in NCCL builds to eight devices. 12 visible GPUs aborted with `peer mapping resources exhausted` (8 and 9 GPUs ran unpatched); 12× Tesla M60 now load `qwen3.6:35b` in one instance (42/42 layers, `-c 262144`, 10.4 tok/s decode).
- **Kolibri-1:** `patch-llama-kolibri1.py` adds the `kolibri1` architecture (runtime part of the patch published with `Hob-forge/Kolibri-1-GGUF`). `Kolibri-1-Q4_K_M` (78.1B MoE, 47.5 GB) loads in one instance over 12× Tesla M60: 51/51 layers, `-c 262144`, batch 64, all in VRAM (53.2 GB).
- **Kolibri-1 measurements:** 8–12 GPUs 10.4–11.1 tok/s decode and 65–69 tok/s prefill; greedy fill (`OLLAMA_FORCE_GPU_LAYERS=1`, scale 1.10) uses 8 of 12 GPUs, 7 fail with CUDA out of memory; needle tests pass up to 250k prompt tokens (250,112 tokens, 180 min, decode 2.7 tok/s); 60-minute soak without errors. Details in [docs/TUNING.md](docs/TUNING.md), raw data in `docs/evidence/kolibri-n02-2026-10-07/`.
- **CI:** `ignore-error=true` on the GitHub Actions cache export, so a lost cache entry does not fail the build.

Details: [docs/FORK-VS-STOCK.md](docs/FORK-VS-STOCK.md), [docs/TUNING.md](docs/TUNING.md), [docs/FLEET-STATE.md](docs/FLEET-STATE.md).

## 2026-10-05

- **Based on:** Ollama v0.35.1 (llama.cpp b11232)
- **Image:** `ollama-gaps:pipefix-20261005` (`cuda12-maxwell`, CUDA 12.0.1, archs 50;52;61;75;86, FA=ON, `BONSAI=OFF`)

### Added

- **GPU coverage:** `cuda12-maxwell` builds 50…90 (Maxwell to Hopper) so that mixed hosts can use one image. The images write their
  targets to `/usr/lib/ollama/CUDA_ARCHS`; `gpu-detect.sh` masks uncovered GPUs (`OLLAMA_UNSUPPORTED_GPU=mask|fail|ignore`),
  and the entrypoint exits with code 3 if no GPU matches. `patch-ollama-discovery.py` determines the compute capability by
  CUDA index and skips devices with an unknown CC (`OLLAMA_ALLOW_UNKNOWN_CC=1` allows them).
- **MTP guard:** `patch-ollama-mtp-default.py` – `OLLAMA_DRAFT_NUM_PREDICT` is the server-wide default and upper bound for
  `draft_num_predict` from the model manifest (0 = MTP off); a request value takes precedence; `gpu-detect.sh` sets 0.
  Go test `TestDraftNumPredictServerDefault`.
- **Embedding in VRAM:** `patch-llama-input-gpu.py` places the input layer (token embedding) on the device of layer 0
  (`LLAMA_INPUT_LAYER_GPU=0` = upstream placement). Recommended with `LLAMA_ARG_FIT_TARGET=256`.
- **Pipeline parallelism:** `patch-llama-pipeline-parallel.py` – `LLAMA_PIPELINE_PARALLEL=0` switches it off (smaller pinned
  host buffers); unchanged without the variable.
- **CUDA graphs on Maxwell:** `patch-llama-cuda-graphs-legacy.py` – opt-in `GGML_CUDA_GRAPHS_LEGACY=1`.
- **Measurement tools:** `scripts/bench-throughput.sh` (effective values from the runner log), `scripts/sweep-batch.sh`,
  `scripts/compare-batch-quality.py`; data in `docs/evidence/batch-quality-2026-10-05/`.
- **CI:** `BONSAI` defaults to `OFF` (repository variable `OLLAMA_BONSAI` or dispatch input `bonsai`); the Spark compat layer
  runs only with `BONSAI=ON`. Bonsai fails on 0.35.1 at the compat patch `002-clef.patch` of the Prism tree.

### Changed

- `gpu-detect.sh` exports `OLLAMA_DRAFT_NUM_PREDICT=0` for all hosts (MTP is slower on the RTX group as well).
- `patch-ollama-dynamic-pool.py` halves `-c` and `-np` when `OLLAMA_MAX_BATCH_SIZE` is set only if Flash Attention
  is explicitly off; the context stays as configured (262144).

### Fixed

- **Fit (nextn slot):** `patch-llama-fit-nextn.py` always counts the nextn slot; with MTP off, layer 0 would otherwise stay on the CPU (`41/42`).
  All instances load `42/42`.

### Hardware state

`qwen3.6:35b` (registry state `a7eb95c53bcf`, 35.5B) runs on N04-RTX (3 GPUs), N02-M60 and N11-M10 with context 262144, q4_0 KV and
42/42 layers; measurements in [docs/TUNING.md](docs/TUNING.md). Open: the cause of the remaining CPU binding of the main thread on Maxwell.

---

## [v0.30.0] — 2026-06-25

**Based on:** Ollama v0.30.10  
**Docker image:** `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`  
**Hardware:** N04-RTX — 4×Tesla M10 · 2×Tesla M60 · GTX 1060 · 3×RTX 2060 · 2×RTX 3060 (~114 GiB VRAM)

### Added

- **Flash Attention on all GPU architectures** (`scripts/patch-ollama-dynamic-pool.py`)  
  Discovery: `ggml_cuda_get_best_fattn_kernel()` returns `BEST_FATTN_KERNEL_TILE` for CC < 7.0  
  (not `NONE`). The TILE kernel runs on Maxwell/Pascal without tensor cores.  
  Result: FA=ON across all 12 GPUs — compute buffer 22–278 MiB instead of 11.4 GiB.

- **Partial-fill greedy fallback** (`scripts/patch-llama-tier-fitting.py`)  
  When greedy fill cannot place all layers, the previous code discarded all placements  
  and redistributed equally. New behaviour: keep greedy-placed layers, distribute only  
  overflow by remaining bandwidth-weighted budget. RTX 3060 retains its 9 greedy  
  layers; Tesla absorbs the overflow — correct pipeline order without OOM.

- **Dynamic GPU pool selection** (`scripts/patch-ollama-dynamic-pool.py`)  
  `selectGPUPool()` on every model load: small models → RTX fast pool (FA=ON, MMA);  
  large models → all 12 GPUs (FA=ON, TILE for Tesla). Prevents global FA=OFF from  
  a single legacy GPU in `CUDA_VISIBLE_DEVICES`.

- **NVML-based GPU auto-detection** (`scripts/gpu-detect.sh`)  
  Runs at container startup. Outputs `CUDA_VISIBLE_DEVICES` (worst→best bandwidth),  
  `OLLAMA_FAST_GPU_DEVICES`, tier threshold, bandwidth per GPU. No nvidia-smi required.

- **Auto-optimization proxy** (`scripts/ollama-proxy.py`, `scripts/auto-optimize.py`)  
  Transparent HTTP proxy (port 11434 → Ollama 11435). Benchmarks scale values and  
  MTP draft tokens per model; caches optimal config in `/root/.ollama/auto-optimize/`.  
  Result for qwen3.6:35b: scale=1.6, draft=2 → **24.3 tok/s** (vs ~16 default).

- **Layout cache** (`scripts/patch-ollama-dynamic-pool.py`, `scripts/auto-optimize.py`)  
  After a successful model load, measures per-GPU VRAM delta via NVML and writes  
  `--tensor-split` proportions to `/root/.ollama/layout-cache/`. Subsequent loads  
  inject the cached split, bypassing the `common_params_fit_impl` estimation loop.

- **Native CUBIN targets** (`dockerfiles/Dockerfile.cuda12-maxwell`)  
  Compiles with `-real` for CC 5.0–9.0: `50-real;52-real;60-real;61-real;70-real;  
  75-real;80-real;86-real;89-real;90-real`. Eliminates PTX JIT delay on cold start.

### Changed

- **CUDA 12.0.1 base image** — broadest driver compatibility for Maxwell (CC 5.0/5.2).
- **FetchContent fork integration** fixed: `FETCHCONTENT_FULLY_DISCONNECTED=ON` prevents  
  CMake from re-running `git checkout <upstream-sha>` after our `fit.cpp` replacement.  
  (→ now copies only `fit.cpp` from fork, keeping Ollama's bundled llama.cpp API.)
- **Proxy connection timeout** increased from 600s → 1800s: large models (llama4:scout)  
  take ~8 minutes to transfer 62 GiB across 12 GPUs. Short timeouts caused the scheduler  
  to propagate `context canceled`, killing llama-server mid-load.
- **`OLLAMA_NUM_PARALLEL=1`** in `.env`: halves KV-cache per GPU.  
  With partial fill, RTX 3060 at 80%+ model → only 2 GiB free → OOM on 2 GiB KV.  
  `np=1` leaves ~3 GiB headroom per GPU.

### Performance on N04-RTX

| Model | Pool | GPUs | FA | tok/s |
|-------|------|------|----|-------|
| qwen3.6:35b Q4\_K\_M | RTX fast | 3–4 of 12 | MMA (CC 7.5+) | **~24.3** (MTP draft=2) |
| llama4:scout Q4\_K\_M | Full 12-GPU | all 12 | MMA + TILE | **~1.7** |

*llama4:scout is bottlenecked by MoE inter-GPU communication over PCIe (no NVLink).  
Each decode step requires transferring active expert weights across 49 layers.*

### Known limitations

- BW-weighted overflow fix (`944fb9a`) in `patch-llama-tier-fitting.py` is not compiled  
  into the binary due to Docker PATCH_GUARD caching: the Python patch sees "Already patched"  
  on rebuild and skips re-application. A `--no-cache` CI build or PATCH_GUARD redesign  
  is needed to activate the fix. Current runtime uses raw-VRAM overflow distribution.
- `llama4:scout` + `qwen3.6:35b` cannot be loaded simultaneously: combined footprint  
  exceeds 114 GiB when KV-cache is included.

---

## [Unreleased]

- BW-weighted partial fill overflow (properly compiled via `--no-cache` build)
- `OLLAMA_NUM_PARALLEL` dynamic selection per pool (np=2 for fast pool, np=1 for full)
- Layout cache validation and cache-invalidation on model update
