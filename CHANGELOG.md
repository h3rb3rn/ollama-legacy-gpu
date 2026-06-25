# Changelog

All notable changes to this fork are documented here.

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
