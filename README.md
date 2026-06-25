# ollama-legacy-gpu

> **Latest release: [v0.30.0](https://github.com/h3rb3rn/ollama-legacy-gpu/releases/tag/v0.30.0)** — based on Ollama v0.30.10 · Docker: `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`

A fork of [Ollama](https://github.com/ollama/ollama) — the Go runtime and Docker packaging — optimized for **heterogeneous multi-GPU pools** that combine modern RTX cards with legacy Tesla M10/M60 GPUs on CUDA 12.

> **Reference system: N04-RTX**  
> AMD EPYC 3151 4-Core · 128 GiB RAM · Ubuntu 22.04 LTS · CUDA 12.0.1 driver  
> 12 GPU endpoints · ~114 GiB VRAM total · No NVLink (PCIe only)

---

## Why this fork exists

Standard `ollama/ollama:latest` targets modern CUDA architectures and drops support for **Tesla M10 (CC 5.0) and M60 (CC 5.2)** — Maxwell-generation data-center GPUs still holding 54 GiB of useful VRAM on N04-RTX. Beyond driver support, upstream Ollama also distributes model layers equally across all GPUs, ignoring their vastly different memory bandwidths (83 GB/s for M10 vs 360 GB/s for RTX 3060).

This fork solves three interconnected problems:

1. **Maxwell GPU support** under CUDA 12 (CC 5.0/5.2 excluded from official CUDA 12)
2. **Bandwidth-aware layer distribution** — potent GPUs fill first; slow GPUs only used as overflow
3. **Per-model GPU pool selection** — fewer GPUs when the model fits; full pool only when needed

---

## Core optimization principle: fill fast GPUs first, use only what you need

The central insight driving all optimizations in this fork:

> **Pipeline throughput is bounded by the slowest GPU. Fewer GPUs in the pipeline means fewer synchronization points and a faster bottleneck.**

On N04-RTX, memory bandwidth spans a 4.3× range:

| GPU class | Bandwidth | Available VRAM |
|-----------|-----------|----------------|
| RTX 3060 (×2) | 360 GB/s | 24 GiB |
| RTX 2060 12GB (×3) | 336 GB/s | 36 GiB |
| GTX 1060 6GB | 192 GB/s | 6 GiB |
| Tesla M60 (×2) | 160 GB/s | ~15 GiB |
| Tesla M10 (×4) | 83 GB/s | 32 GiB |

If you spread `qwen3.6:35b` (22 GiB) across all 12 GPUs, each gets ~1.8 GiB — including four Tesla M10s at 83 GB/s. Every decode step then synchronizes across all 12 GPUs and waits for the slowest one. Measured result: ~4 tok/s.

With greedy fill (RTX only, 4 GPUs): **24.3 tok/s with speculative decoding** — a 6× improvement on the same hardware.

The principle: **assign layers greedily to the fastest GPUs, stop when all layers are placed.**

---

## Reference Hardware: N04-RTX

**Host system:**
- CPU: AMD EPYC 3151 4-Core Processor
- RAM: 128 GiB DDR4 ECC
- OS: Ubuntu 22.04.5 LTS
- CUDA driver: 12.0.1 (no NVLink — all inter-GPU communication over PCIe)

**GPU topology (CUDA order, worst → best bandwidth):**

| CUDA | GPU | Arch | CC | VRAM | Bandwidth |
|------|-----|------|----|------|-----------|
| 0–3 | Tesla M10 (×4) | Maxwell | 5.0 | 8 GiB each | 83 GB/s |
| 4–5 | Tesla M60 (×2) | Maxwell | 5.2 | ~7.7 GiB each | 160 GB/s |
| 6 | GTX 1060 6GB | Pascal | 6.1 | 6 GiB | 192 GB/s |
| 7–9 | RTX 2060 12GB (×3) | Turing | 7.5 | 12 GiB each | 336 GB/s |
| 10–11 | RTX 3060 (×2) | Ampere | 8.6 | 12 GiB each | 360 GB/s |

**Total: ~114 GiB across 12 GPU dies on a single PCIe host.**

CUDA ordering (worst → best) is intentional: the greedy fill algorithm fills from
CUDA 11 (RTX 3060) downward, exhausting fast GPUs before touching slow ones.

---

## Key Changes vs upstream Ollama

### 1. Dynamic GPU Pool Selection

**File:** `scripts/patch-ollama-dynamic-pool.py` → patches `llm/llama_server.go`

On every model load, `selectGPUPool()` checks the model file size against the RTX-only
pool capacity and routes accordingly:

```
Model ≤ 75% of RTX fast pool (~48 GiB threshold):
  → CUDA_VISIBLE_DEVICES = 5 RTX GPUs only (64 GiB)
  → Greedy fill: fills RTX 3060 → RTX 2060, stops when done
  → Flash Attention ON (CC ≥ 7.5, MMA kernel)
  → Result: 3–4 GPUs used, Tesla untouched, maximum tok/s

Model > threshold:
  → CUDA_VISIBLE_DEVICES = all 12 GPUs (114 GiB)
  → Greedy fill with bandwidth weighting (see below)
  → Flash Attention ON — TILE kernel handles Maxwell/Pascal (see section 4)
  → Result: RTX fills first, Tesla/GTX only for overflow capacity
```

**Why pool selection matters — and why this is per-model, not global:**

In standard Ollama, a single Tesla M10 (CC 5.0) in `CUDA_VISIBLE_DEVICES` forces
Flash Attention OFF for the entire server process — affecting all models, including
those that would never touch a Tesla GPU. This is a global, permanent flag in upstream.

This fork makes it **per-model and dynamic**:

- Model fits in RTX pool → `CUDA_VISIBLE_DEVICES` restricted to RTX UUIDs → FA=ON (MMA kernel)
- Model requires full pool → all 12 GPUs → FA=ON via TILE kernel for Maxwell/Pascal

Whether FA is active depends entirely on **which GPUs the specific model actually uses**,
not on which GPUs are installed. A server running both `qwen3.6:35b` (RTX pool, FA=MMA)
and `llama4:scout` (full pool, FA=TILE) runs both with Flash Attention enabled
simultaneously — each with the kernel appropriate for its assigned GPUs.

The disabling of modern features like FA is **never permanent** in this fork; it is
only an upstream limitation that this project removes through pool-aware routing.

### 2. Greedy Fill with Bandwidth Weighting

**File:** `scripts/patch-llama-tier-fitting.py` → patches `common/fit.cpp`  
**Native C++ version:** `h3rb3rn/llama.cpp-legacy-gpu` (branch `legacy-gpu-support`)

Standard llama.cpp distributes layers across all visible GPUs proportionally to their
VRAM. For a 22 GiB model across 12 GPUs, every GPU including Tesla M10 gets the same
share. Decode throughput collapses to the M10's 83 GB/s bottleneck.

**Greedy fill algorithm:**

```
For each GPU sorted by bandwidth (best → worst, CUDA 11 → 0):
    effective_budget = (free_vram − margins) × (this_bw / max_bw)
    n_layers = effective_budget / bytes_per_layer
    assign layers; reduce remaining
    stop if all layers placed

If greedy cannot place all layers (tight overhead):
    fall back to VRAM-weighted distribution across all GPUs
    (proportional to free_vram − margins, not equal shares)
```

The bandwidth factor is critical: Tesla M10 (83 GB/s) gets an effective budget of
`8 GiB / 4.3 = 1.9 GiB`, while RTX 3060 (360 GB/s) gets its full 12 GiB.
This prevents assigning many layers to a GPU that would become a pipeline bottleneck.

**Results on N04-RTX:**

| Model | Layers | GPUs actually used | Unused GPUs | tok/s |
|-------|--------|-------------------|-------------|-------|
| qwen3.6:35b (22 GiB) | 42 | 3–4 RTX | 8 Tesla/GTX/RTX | ~24.3 |
| llama4:scout (62 GiB) | 49 | all 12 | none | ~1 |

For qwen: **8 GPUs stay idle** because RTX cards can hold the entire model. For
llama4:scout: all 12 are needed because the model exceeds the RTX pool (60 GiB).

### 3. Flash Attention on All Architectures (including Maxwell)

**Discovery:** `ggml-cuda/fattn.cu` in llama.cpp dispatches FA kernels by compute
capability at runtime:

| CC | GPU class | FA kernel used | Compute buffer |
|----|-----------|---------------|----------------|
| ≥ 8.6 | RTX 3060 | `MMA_F16` (Ampere tensor cores) | ~76 MiB |
| ≥ 7.5 | RTX 2060 | `MMA_F16` (Turing tensor cores) | ~76 MiB |
| ≥ 7.0 | Volta | `WMMA_F16` | ~76 MiB |
| ≥ 5.0 | Tesla M10/M60, GTX 1060 | **`TILE`** (generic CUDA cores) | ~76 MiB |

The `TILE` kernel requires no tensor cores — it is the generic FA fallback that runs
on any CUDA architecture ≥ CC 5.0. `BEST_FATTN_KERNEL_NONE` (abort) is never returned
for CC ≥ 5.0 in our build.

**Impact:** Flash Attention ON across all 12 GPUs including Tesla M10. The primary GPU's
compute buffer drops from **11,444 MiB → 76 MiB**, enabling large models at 131K+
context without OOM on any GPU in the pool.

### 4. Native CUBIN Targets (No PTX JIT)

Standard builds use `-virtual` CUDA targets for older architectures, causing JIT
compilation of PTX bytecode on first model load. With 12 GPUs and PTX for CC 5.0:
cold-start delay of 20–30 minutes.

This build compiles native CUBIN for every target:
```
50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real
```

Kernels load instantly on all 12 GPUs from the first request.

### 5. Auto-Optimization Proxy

**Files:** `scripts/ollama-proxy.py`, `scripts/auto-optimize.py`

A transparent HTTP proxy (port 11434 → Ollama on 11435) that:

1. Intercepts the first request to any model
2. Launches a background optimizer testing:
   - `OVERHEAD_SCALE` ∈ [1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
   - MTP speculative decoding draft tokens [0, 2, 4] at each scale
3. Caches the globally optimal `(scale, draft)` pair in `/root/.ollama/auto-optimize/`
4. Applies cached settings on every subsequent load

The optimizer tests MTP at all scale values because GPU count affects speculative
decoding efficiency: with 4 GPUs (scale=1.6) each GPU has fewer layers per pass,
making draft tokens worth the overhead. With 3 GPUs (scale=1.1), they degrade throughput.

**Proxy connection timeout:** 1800s to accommodate large model load times (llama4:scout
requires ~8 minutes to transfer 62 GiB across 12 GPUs). Shorter timeouts cause the
scheduler to cancel loading and retry indefinitely.

### 6. Persistent Layout Cache

**Files:** `scripts/patch-ollama-dynamic-pool.py`, `scripts/auto-optimize.py`, `scripts/ollama-proxy.py`

After a successful model load, `auto-optimize.py` measures per-GPU VRAM delta via NVML
and derives the `--tensor-split` proportions used by llama.cpp. These are written to
`/root/.ollama/layout-cache/<model_sha>-<gpu_count>.split` (persistent volume).

On the next restart, `selectGPUPool()` reads the cache and injects `--tensor-split`
directly, bypassing the `common_params_fit_impl` estimation loop (20–30 iterations
visible in logs). Cache key encodes model SHA and GPU count so fast-pool and full-pool
layouts are stored separately.

---

## Model Performance on N04-RTX

| Model | Size | Pool | GPUs used | Flash Attn | tok/s |
|-------|------|------|-----------|------------|-------|
| qwen3.6:35b Q4_K_M | 22 GiB | RTX fast (5 avail.) | **3–4 RTX** | MMA (CC 7.5+) | **~24.3** (with MTP draft=2, scale=1.6) |
| llama4:scout Q4_K_M | 62 GiB | Full 12-GPU | **all 12** | MMA + TILE | **~1** |

**Why llama4:scout is slow (~1 tok/s) even with all 12 GPUs on GPU:**
llama4:scout uses a Mixture-of-Experts (MoE) architecture: 16 expert FFN blocks per
layer with only 1 active per token. The active expert's weights can reside on any GPU,
requiring PCIe transfers between GPUs on every decode step. Without NVLink (which
provides 600 GB/s vs PCIe's ~32 GB/s per lane), each MoE decode step crosses the PCIe
bus 49 times (once per layer). This is a fundamental hardware constraint — not a
software problem. llama4:scout is designed for systems with NVLink or single GPUs
with ≥ 80 GiB VRAM (A100/H100).

**qwen3.6:35b** is a dense transformer (no MoE) and benefits fully from the greedy
fill approach: all active layers reside on high-bandwidth RTX GPUs with no slow-GPU
bottleneck.

---

## Quick Start

```bash
# On N04-RTX: pull and start
cd /opt/deployment/ollama/fork/compose
docker compose -f docker-compose.worker-rtx.yml pull
docker compose -f docker-compose.worker-rtx.yml up -d

# Or build from source
docker build \
  -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.9.0 \
  --build-arg LLAMA_CPP_FORK=https://github.com/h3rb3rn/llama.cpp-legacy-gpu.git \
  -t ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest .
```

**Environment variables (set automatically by `gpu-detect.sh` via NVML):**

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU order: worst→best bandwidth (greedy fill direction) |
| `OLLAMA_FAST_GPU_DEVICES` | RTX-only UUIDs for fast pool |
| `OLLAMA_FAST_POOL_VRAM_GB` | Total RTX VRAM (threshold for pool selection) |
| `OLLAMA_GPU_TIER_THRESHOLD` | CUDA index split: legacy vs fast GPUs |
| `OLLAMA_GPU_BANDWIDTHS` | Per-GPU bandwidth in GB/s (for layer budget weighting) |

---

## Related Projects

- **Upstream Ollama**: https://github.com/ollama/ollama (MIT)
- **llama.cpp fork**: https://github.com/h3rb3rn/llama.cpp-legacy-gpu (MIT)
- **Upstream llama.cpp**: https://github.com/ggml-org/llama.cpp (MIT)

---

## License

MIT License — same as upstream Ollama and llama.cpp.

Modifications Copyright (c) 2025–2026 Philipp Horn.  
Original Ollama code Copyright (c) Ollama contributors.  
See [LICENSE](LICENSE) for the full MIT license text.
