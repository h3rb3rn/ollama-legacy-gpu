# ollama-legacy-gpu

A fork of [Ollama](https://github.com/ollama/ollama) — the Go runtime and Docker packaging — optimized for **heterogeneous multi-GPU pools** that combine modern RTX cards with legacy Tesla M10/M60 GPUs on CUDA 12.

> **Reference system: N04-RTX**  
> AMD EPYC 3151 4-Core · 128 GiB RAM · Ubuntu 22.04 LTS · CUDA 12.0.1 driver  
> 12 GPU endpoints · ~114 GiB VRAM total · No NVLink (PCIe only)

---

## Why this fork exists

Standard `ollama/ollama:latest` targets modern CUDA architectures (CC ≥ 6.0 with practical support for CC ≥ 7.5). **Tesla M10 (CC 5.0) and M60 (CC 5.2)** are Maxwell-generation GPUs released in 2015 — still fully functional, but excluded from official CUDA toolkit distributions starting with CUDA 12.

This project solves a real hardware problem: running state-of-the-art LLMs on a mixed GPU server during a hardware supply shortage, where legacy data-center GPUs (Tesla M10/M60) are combined with consumer RTX cards — all in a single Ollama instance sharing the full ~114 GiB VRAM pool.

This fork enables:
1. **Single Ollama instance across all 12 GPU endpoints** (~114 GiB VRAM)
2. **Dynamic GPU pool selection** per model: fast RTX pool for small models, full 12-GPU pool for large models
3. **Greedy layer fill** instead of equal distribution — RTX cards get filled first, Tesla only used as overflow
4. **Bandwidth-weighted budgets** — slow Tesla GPUs receive proportionally fewer layers to prevent pipeline bottlenecks
5. **Tier-aware Flash Attention** — FA enabled when the model uses only RTX GPUs (CC ≥ 7.5)
6. **Auto-optimization proxy** — background optimizer finds optimal settings per model and caches them

---

## Reference Hardware: N04-RTX

**Host system:**
- CPU: AMD EPYC 3151 4-Core Processor
- RAM: 128 GiB DDR4 ECC
- OS: Ubuntu 22.04.5 LTS (kernel 5.15.0-181)
- CUDA driver: 12.0.1 (no NVLink — all GPUs communicate over PCIe)

**GPU topology (CUDA order, worst → best bandwidth):**

| CUDA idx | GPU | Architecture | CC | VRAM | Bandwidth | PCIe ID |
|----------|-----|-------------|-----|------|-----------|---------|
| 0–3 | Tesla M10 (4 GPU dies) | Maxwell | 5.0 | 8 GiB each | 83 GB/s | 13–16:00.0 |
| 4–5 | Tesla M60 (2 GPU dies) | Maxwell | 5.2 | ~7.7 GiB each | 160 GB/s | 0d–0e:00.0 |
| 6 | GeForce GTX 1060 6GB | Pascal | 6.1 | 6 GiB | 192 GB/s | 17:00.0 |
| 7–9 | GeForce RTX 2060 12GB (×3) | Turing | 7.5 | 12 GiB each | 336 GB/s | 07,0a,0f:00.0 |
| 10 | GeForce RTX 3060 LHR | Ampere | 8.6 | 12 GiB | 360 GB/s | 08:00.0 |
| 11 | GeForce RTX 3060 | Ampere | 8.6 | 12 GiB | 360 GB/s | 09:00.0 |

**Total VRAM: ~114 GiB across 12 GPU dies on a single PCIe host.**

The CUDA ordering (worst → best) is critical: our greedy fill starts from CUDA 11
(RTX 3060, 360 GB/s) and works backward, filling each GPU completely before extending
to slower GPUs. Tesla M10 only receives layers if RTX + GTX cannot hold the model.

---

## Key Changes vs upstream Ollama

### 1. Dynamic GPU Pool Selection

**File:** `scripts/patch-ollama-dynamic-pool.py` → patches `llm/llama_server.go`

Adds `selectGPUPool(*llamaServerLaunchConfig)` called on every model load:

```
Model file size ≤ 75% of RTX fast pool (64 GiB):
  → CUDA_VISIBLE_DEVICES = 5 RTX UUIDs only
  → OLLAMA_FORCE_GPU_LAYERS=999 (greedy fill, see below)
  → Flash Attention ON (all RTX ≥ CC 7.5)
  → Example: qwen3.6:35b (22 GiB) → 4 RTX GPUs, ~24.3 tok/s with MTP

Model file size > threshold:
  → All 12 GPUs (114 GiB full pool)
  → Standard Ollama fitting (compute-buffer-aware, prevents OOM on M10)
  → Flash Attention OFF (Tesla cannot run FA kernels)
  → Example: llama4:scout (62 GiB) → RTX + GTX + M60 + M10 as needed
```

**Why this matters:** Without pool selection, Tesla GPUs are always visible to Ollama's
Flash Attention check. One CC 5.0 GPU disables FA globally — even for a model that
could run entirely on RTX. This unnecessarily increases KV cache size by 4× (f16
instead of q4_0) and reduces throughput.

### 2. Greedy Fill with Bandwidth Weighting

**File:** `scripts/patch-llama-tier-fitting.py` → patches `common/fit.cpp`  
**Native C++ version:** `h3rb3rn/llama.cpp-legacy-gpu` (see below)

Standard llama.cpp distributes layers across GPUs proportionally to VRAM. For a
22 GiB model across 12 GPUs, each GPU gets ~1.8 GiB of layers — including Tesla M10
(83 GB/s). The decode pipeline then waits at the slowest GPU: throughput collapses.

Our greedy fill with bandwidth weighting:
```
for each GPU (best → worst bandwidth, CUDA 11 → 0):
    effective_budget = (free_vram - margins) / bandwidth_factor
    bandwidth_factor = max_bw / this_gpu_bw  (M10: 360/83 = 4.3×)
    n_layers = effective_budget / bytes_per_layer
    assign n_layers; stop when all layers placed
```

**Impact on N04-RTX:**
- `qwen3.6:35b` (22 GiB, 42 layers): 3–4 RTX GPUs used, Tesla untouched
- `llama4:scout` (62 GiB, 49 layers): RTX → GTX → M60 → M10 in order, only as needed

Fewer GPUs in the pipeline = fewer PCIe synchronization points = higher tok/s.

### 3. VRAM-Weighted Fallback (replaces equal distribution)

When greedy fill cannot place all layers (scale too tight), the original equal
distribution (`tensor_split = [1, 1, 1, ...]`) caused OOM on Tesla M10 for large
models. CUDA 0 (M10) as the primary orchestration device needs a large compute buffer
for attention (scales with `n_ctx × n_heads × 4 bytes`). Equal distribution assigned
Tesla M10 as many layers as RTX 3060 despite having 8 GiB vs 12 GiB and 4× lower
bandwidth.

**Fix:** VRAM-weighted fallback — each GPU's share proportional to
`(free_vram - margins) / total_budget`. Tesla M10 gets 1-2 layers; RTX 3060 gets 8.

### 4. Tier-Aware Flash Attention

**File:** `scripts/patch-ollama-fa.py` → patches `LlamaServerFlashAttention()` in Go

When `OLLAMA_GPU_TIER_THRESHOLD > 0`, FA is checked only against the fast GPU tier
(`gpus[threshold:]` = RTX GPUs). A Tesla M10 in the visible device list no longer
forces global FA=OFF.

**Impact:** Fast-pool models (small enough for RTX only) get FA=ON → q4_0 KV cache
→ 4× smaller KV footprint → more context fits in RTX VRAM.

### 5. Auto-Optimization Proxy

**Files:** `scripts/ollama-proxy.py`, `scripts/auto-optimize.py`

A transparent HTTP proxy on port 11434 (Ollama internally on 11435) that:

1. Detects each new model on first request
2. Spawns a background optimizer testing `OVERHEAD_SCALE` ∈ [1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
   and MTP draft tokens [0, 2, 4] at each scale
3. Caches the globally optimal `(scale, draft)` combination in `/root/.ollama/auto-optimize/`
4. Applies cached settings on all subsequent loads

**Key insight:** The optimizer tests MTP at ALL scale values, not just the best
no-MTP scale. This is important because with more GPUs (higher scale), each GPU has
fewer layers and shorter per-pass time, which can make speculative decoding more or
less effective.

**Benchmark results on N04-RTX for `qwen3.6:35b` Q4_K_M:**

| Scale | GPUs | tok/s (no draft) | tok/s (draft=2) |
|-------|------|-----------------|-----------------|
| 1.1 | 3 RTX | 18.0 | worse |
| 1.6 | 4 RTX | 16.2 | **24.3** ← optimal |

MTP improves throughput at scale=1.6 (4 GPUs) but degrades it at scale=1.1 (3 GPUs)
because the draft overhead outweighs the benefit when each GPU has many layers.

### 6. Native CUBIN Targets (No PTX JIT)

Standard builds use `-virtual` CUDA targets for older architectures, causing
per-GPU JIT compilation on first model load. With 12 GPUs and PTX for CC 5.0:
startup delay of 20-30 minutes on cold start.

This build uses `-real` for all architectures:
```
50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real
```

Native CUBIN embedded → instant kernel loading on all 12 GPUs.

---

## Model Performance on N04-RTX

| Model | Weights | Pool | GPUs | CPU/GPU | tok/s |
|-------|---------|------|------|---------|-------|
| qwen3.6:35b Q4_K_M | 22 GiB | RTX fast (5 RTX) | 4/12 | ~7%/93% | ~24.3 |
| llama4:scout Q4_K_M | 62 GiB | Full 12-GPU | 10-11/12 | ~8%/92% | ~2-5 |

*The ~7% CPU share reflects the model-level embedding layer (architecturally on CPU in
llama.cpp for MoE models) and PCIe graph coordination overhead — not actual layer
compute on CPU.*

*NVLink would eliminate the PCIe synchronization overhead and likely double throughput,
but is not available on this consumer/prosumer hardware mix.*

---

## Quick Start

```bash
# On N04-RTX: pull and start
cd /opt/deployment/ollama/fork/compose
./update.sh

# Or build from source
docker build \
  -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.9.0 \
  --build-arg LLAMA_CPP_FORK=https://github.com/h3rb3rn/llama.cpp-legacy-gpu.git \
  -t ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest .
```

---

## Related Projects

- **Upstream Ollama**: https://github.com/ollama/ollama (MIT)
- **llama.cpp fork**: https://github.com/h3rb3rn/llama.cpp-legacy-gpu (MIT)
- **Upstream llama.cpp**: https://github.com/ggml-org/llama.cpp (MIT)
- **Inspiration** — DwarfStar's chunked-prefill approach: https://github.com/antirez/dwarfstar

---

## License

MIT License — same as upstream Ollama and llama.cpp.

Modifications in this repository are Copyright (c) 2025-2026 Philipp Horn.  
Original Ollama code is Copyright (c) Ollama contributors.  
See [LICENSE](LICENSE) for the full MIT license text.
