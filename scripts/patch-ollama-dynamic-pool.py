#!/usr/bin/env python3
"""
patch-ollama-dynamic-pool.py — Patches llm/llama_server.go to dynamically
select GPU pool (fast-only or all GPUs) per model load based on model size.

Behavior:
  If model file size <= OLLAMA_FAST_POOL_VRAM_GB × 0.75 × 1 GB:
    → Use only fast GPUs (OLLAMA_FAST_GPU_DEVICES)
    → Enable Flash Attention (OLLAMA_FLASH_ATTENTION=1)
    → Small models (≤60 GB): 50-60 tok/s on RTX pool

  If model file size > threshold OR env vars not set:
    → Use all 12 GPUs (no CUDA_VISIBLE_DEVICES override)
    → Force FA=ON: RTX uses MMA kernel, Tesla/GTX use TILE kernel (no tensor cores)
    → Greedy fill (FORCE_GPU_LAYERS=999): RTX fills first, Tesla last
    → Large models (>60 GB): full 114 GB pool, no CPU fallback for llama4:scout

Required env vars (set by gpu-detect.sh):
  OLLAMA_FAST_GPU_DEVICES     Comma-separated UUIDs of fast GPUs (CC >= 7.5)
  OLLAMA_FAST_POOL_VRAM_GB    Total VRAM of fast pool in GB (integer)

Example (N04-RTX with 5 RTX GPUs = 58 GB):
  OLLAMA_FAST_GPU_DEVICES=GPU-ed954d67,...,GPU-63bfbd4b
  OLLAMA_FAST_POOL_VRAM_GB=58

The patch adds a function selectGPUPool() that is called before
appendFlashAttentionArgs() for each model launch.
"""

import sys
import re
from pathlib import Path

PATCH_GUARD   = "OLLAMA_FAST_POOL_VRAM_GB"
TARGET_FILE   = "llm/llama_server.go"
INSERT_BEFORE = "params = appendFlashAttentionArgs(params, launch.gpus)"


POOL_SELECT_FUNC = '''
// Imports needed by selectGPUPool (strings for TrimSpace on override file)
// Note: "strings", "os", "strconv" are already imported in llama_server.go.

// ── Layout cache helpers ──────────────────────────────────────────────────────
// Persists the successful --tensor-split from a previous model load so the
// llama.cpp fitting algorithm (common_params_fit_impl) can be bypassed on
// subsequent loads of the same model on the same GPU pool.
//
// Cache key  : first 16 chars of model blob SHA + GPU count (pool discriminator).
// Cache file : /root/.ollama/layout-cache/<key>.split  (persistent volume).
// Cache value: ":"-separated integer proportions per CUDA device (0 = no layers).
//              Written by auto-optimize.py after measuring per-GPU VRAM delta.
// Lifetime   : no TTL — valid as long as the model blob exists in the pool.
//              Invalidated automatically when CUDA_VISIBLE_DEVICES count changes
//              (different pool) or model SHA changes (different model/quant).

func _layoutCacheKey(modelPath, cudaVis string) string {
	base := modelPath
	if idx := strings.LastIndex(modelPath, "/"); idx >= 0 {
		base = modelPath[idx+1:]
	}
	// Strip "sha256-" prefix to get the raw hash
	if strings.HasPrefix(base, "sha256-") {
		base = base[7:]
	}
	if len(base) > 16 {
		base = base[:16]
	}
	// GPU count discriminates fast pool (5 RTX) from full pool (12 GPUs)
	gpuCount := strings.Count(cudaVis, ",") + 1
	return base + "-" + strconv.Itoa(gpuCount)
}

func _readLayoutCache(modelPath, cudaVis string) string {
	key := _layoutCacheKey(modelPath, cudaVis)
	data, err := os.ReadFile("/root/.ollama/layout-cache/" + key + ".split")
	if err != nil {
		return ""
	}
	split := strings.TrimSpace(string(data))
	if split == "" || !strings.Contains(split, ":") {
		return ""
	}
	slog.Info("layout cache hit: will inject --tensor-split", "key", key, "split", split)
	return split
}

func _writeLayoutKey(modelPath, cudaVis string) {
	key := _layoutCacheKey(modelPath, cudaVis)
	_ = os.WriteFile("/tmp/ollama-layout-key", []byte(key+"\\n"), 0644)
}

// selectGPUPool dynamically restricts CUDA_VISIBLE_DEVICES to the fast GPU pool
// (OLLAMA_FAST_GPU_DEVICES) when the model file fits within the fast pool capacity.
// This enables Flash Attention on the fast-only pool while still allowing large
// models to use the full GPU pool (including legacy Tesla GPUs) when needed.
//
// Env vars (set by gpu-detect.sh):
//   OLLAMA_FAST_GPU_DEVICES    comma-separated UUID list of fast GPUs (CC >= 7.5)
//   OLLAMA_FAST_POOL_VRAM_GB   total fast pool VRAM in GB
//
// Dynamic override (written by auto-optimize.py between model loads):
//   /tmp/ollama-scale-override  contains a float OVERHEAD_SCALE value
//   Read on every call so the auto-optimizer can update between loads.
func selectGPUPool(launch *llamaServerLaunchConfig) {
	// Apply auto-optimizer scale override if present
	const scaleOverrideFile = "/tmp/ollama-scale-override"
	if data, err := os.ReadFile(scaleOverrideFile); err == nil {
		if scale := strings.TrimSpace(string(data)); scale != "" {
			os.Setenv("OLLAMA_LAYER_OVERHEAD_SCALE", scale)
		}
	}

	fastDevices := os.Getenv("OLLAMA_FAST_GPU_DEVICES")
	fastPoolGB  := os.Getenv("OLLAMA_FAST_POOL_VRAM_GB")
	if fastDevices == "" || fastPoolGB == "" {
		return // auto-detection not configured, keep current GPU set
	}
	poolGB, err := strconv.ParseInt(fastPoolGB, 10, 64)
	if err != nil || poolGB <= 0 {
		return
	}
	// Use 75% of fast pool VRAM as threshold (leaves headroom for KV-cache + compute)
	thresholdBytes := poolGB * 1_000_000_000 * 75 / 100

	// Get model file size as proxy for weight memory
	info, err := os.Stat(launch.modelPath)
	if err != nil || info.Size() <= 0 {
		return
	}
	modelBytes := info.Size()

	if launch.extraEnvs == nil {
		launch.extraEnvs = make(map[string]string)
	}

	if modelBytes <= thresholdBytes {
		// Model fits in fast pool (RTX-only, FA-capable GPUs).
		// Use greedy fill: fill best RTX GPU completely before moving to next.
		// Fewer GPUs in the pipeline = less inter-GPU overhead = higher tok/s.
		// Safe: RTX GPUs support FA → compute buffers are O(seq_len), not O(seq_len²).
		slog.Info("dynamic GPU pool: model fits in fast pool, restricting to fast GPUs",
			"model_gb", modelBytes/(1<<30),
			"pool_gb", poolGB,
			"devices", fastDevices)
		launch.extraEnvs["CUDA_VISIBLE_DEVICES"] = fastDevices
		launch.extraEnvs["OLLAMA_FORCE_GPU_LAYERS"] = "999"
		os.Setenv("OLLAMA_FORCE_GPU_LAYERS", "999")
		// Clear full-pool batch cap (FA makes compute buffers tiny, no cap needed).
		os.Setenv("OLLAMA_MAX_BATCH_SIZE", "")
		// Explicitly enable FA: fast GPUs all support FA (CC >= 7.5)
		os.Setenv("OLLAMA_FLASH_ATTENTION", "true")
		// Cache key uses the EFFECTIVE CUDA_VISIBLE_DEVICES (fast pool = RTX UUIDs).
		if _split := _readLayoutCache(launch.modelPath, fastDevices); _split != "" {
			os.Setenv("OLLAMA_CACHED_TENSOR_SPLIT", _split)
		}
		_writeLayoutKey(launch.modelPath, fastDevices)
	} else {
		// Model exceeds fast pool — use all 12 GPUs with FA=ON forced.
		//
		// Key finding from ggml-cuda/fattn.cu source analysis:
		//   FA dispatch (ggml_cuda_get_best_fattn_kernel) returns BEST_FATTN_KERNEL_TILE
		//   for GPUs without tensor cores (CC < 7.0), NOT BEST_FATTN_KERNEL_NONE.
		//   TILE kernel is the generic FA path requiring only standard CUDA cores.
		//   Compute buffer stays O(tile_size) = tiny (≈ 76–200 MiB) on ALL GPUs.
		//
		//   Device dispatch by compute capability:
		//     Tesla M10 (CC 5.0) → BEST_FATTN_KERNEL_TILE (no tensor cores)
		//     Tesla M60 (CC 5.2) → BEST_FATTN_KERNEL_TILE
		//     GTX 1060  (CC 6.1) → BEST_FATTN_KERNEL_TILE
		//     RTX 2060  (CC 7.5) → BEST_FATTN_KERNEL_MMA_F16 (Turing tensor cores)
		//     RTX 3060  (CC 8.6) → BEST_FATTN_KERNEL_MMA_F16 (Ampere tensor cores)
		//
		// All 12 GPUs (114 GB) with FA=ON eliminates CPU fallback for models up to
		// ~110 GB. llama4:scout (63 GB) fits entirely in GPU VRAM with headroom.
		//
		// CUDA ordering: parent CUDA_VISIBLE_DEVICES (worst-bandwidth → best).
		//   CUDA0  = Tesla M10   (83 GB/s, 8 GB)
		//   CUDA11 = RTX 3060   (360 GB/s, 12 GB)
		// Greedy fill (FORCE_GPU_LAYERS=999) fills CUDA11→CUDA0 (best→worst):
		//   RTX 3060 (2×12 GB = 24 GB) → RTX 2060 (3×12 GB = 36 GB) first,
		//   extending to GTX/Tesla only when RTX is exhausted.
		// Bandwidth weighting (patch-llama-tier-fitting.py) further limits Tesla
		// layer budget (83 GB/s vs 360 GB/s = 4.3× fewer effective layers) to
		// avoid pipeline bottlenecks.
		//
		// No CUDA_VISIBLE_DEVICES override: keep parent ordering (worst→best).
		slog.Info("dynamic GPU pool: model exceeds fast pool, using all 12 GPUs (FA=ON, TILE kernel for CC<7.0)",
			"model_gb", modelBytes/(1<<30),
			"threshold_gb", thresholdBytes/(1<<30),
			"fast_pool_gb", poolGB)
		// Greedy fill: best GPU (RTX, highest CUDA index) fills first.
		os.Setenv("OLLAMA_FORCE_GPU_LAYERS", "999")
		// Disable legacy tier patches: bandwidth weighting handles distribution.
		os.Setenv("OLLAMA_GPU_TIER_THRESHOLD", "0")
		// FA=ON: TILE kernel available on all architectures ≥ CC 5.0.
		// RTX uses MMA FA (fast), Tesla/GTX use TILE FA (slower but tiny compute buffer).
		os.Setenv("OLLAMA_FLASH_ATTENTION", "true")
		// Enable batch cap to reduce -np 2→1, halving the KV-cache per GPU.
		// With partial fill, RTX 3060 gets ~9.8 GiB model; KV at np=2 needs 2 GiB
		// on CUDA10 (leaving only 0.4 GiB margin → OOM). np=1 halves KV to ~1 GiB.
		if os.Getenv("OLLAMA_MAX_BATCH_SIZE") == "" {
			os.Setenv("OLLAMA_MAX_BATCH_SIZE", "64")
		}
		// Tighten overhead scale for full pool with FA=ON.
		//
		// Default scale=1.4 was calibrated for FA=OFF where the gallocr compute buffer
		// is 11.4 GiB on the primary GPU. With FA=ON the compute buffer shrinks to
		// ~278 MiB per GPU — the default 1.4× overhead makes greedy fill abort too
		// early ("overhead=1.30x too tight") and fall back to VRAM-weighted distribution
		// across all 12 GPUs, which spreads layers equally instead of filling RTX first.
		//
		// Scale=1.2: reserves 20% of each GPU's VRAM budget for KV-cache after
		// model weights are placed. This prevents OOM when the KV-cache is allocated
		// after greedy fill (RTX 3060 at 80%+ model → only 2 GiB left → OOM on 2 GiB KV).
		// With FA=ON compute buffers are tiny (~278 MiB), so 1.2 leaves enough headroom
		// for both the partial-fill overflow placement and the KV-cache allocation.
		if cur := os.Getenv("OLLAMA_LAYER_OVERHEAD_SCALE"); cur == "" || cur > "1.2" {
			os.Setenv("OLLAMA_LAYER_OVERHEAD_SCALE", "1.2")
		}
		// Cache key uses the EFFECTIVE CUDA_VISIBLE_DEVICES (full pool = all 12 GPUs).
		_allVis := os.Getenv("CUDA_VISIBLE_DEVICES")
		if _split := _readLayoutCache(launch.modelPath, _allVis); _split != "" {
			os.Setenv("OLLAMA_CACHED_TENSOR_SPLIT", _split)
		}
		_writeLayoutKey(launch.modelPath, _allVis)
	}
}

'''


def find_target(ollama_root: Path) -> Path | None:
    candidates = [ollama_root / TARGET_FILE, ollama_root / "src" / TARGET_FILE]
    for c in candidates:
        if c.is_file():
            return c
    return None


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    if INSERT_BEFORE not in content:
        print(f"  Marker not found: {INSERT_BEFORE!r}", file=sys.stderr)
        return False

    # Add function body before the target line in the file
    # Find a good location — before the first top-level func after package declaration
    # Insert selectGPUPool() function near LlamaServerFlashAttention (our previous patch)
    if "LlamaServerFlashAttention" in content:
        # Insert before the FA function
        insert_marker = "// LlamaServerFlashAttention"
        new_content = content.replace(insert_marker, POOL_SELECT_FUNC + insert_marker, 1)
    else:
        # Fallback: insert before the INSERT_BEFORE line
        new_content = content.replace(
            INSERT_BEFORE,
            "selectGPUPool(&launch)\n\t" + INSERT_BEFORE,
            1
        )
        # This won't have the function definition — needs separate placement
        # Just add the call for now
        print("  Warning: FA function not found, adding call only")

    if new_content == content:
        print(f"  No change made", file=sys.stderr)
        return False

    # Add the call to selectGPUPool + inline batch cap before appendFlashAttentionArgs.
    # params is in scope here; we search for --batch-size / --ubatch-size and
    # replace values > OLLAMA_MAX_BATCH_SIZE. This reduces the per-GPU prefill
    # compute buffer from 11.6 GiB (batch=512) to 1.07 GiB (batch=64), allowing
    # Tesla M10/M60 to participate in llama4:scout inference on all 12 GPUs.
    INLINE_BATCH_CAP = '''selectGPUPool(&launch)
\t// [layout cache: inject cached --tensor-split to bypass llama.cpp fitting algorithm]
\tif _ts := os.Getenv("OLLAMA_CACHED_TENSOR_SPLIT"); _ts != "" {
\t\tparams = append(params, "--tensor-split", _ts)
\t\tos.Unsetenv("OLLAMA_CACHED_TENSOR_SPLIT")
\t\tslog.Info("layout cache: injecting cached --tensor-split", "split", _ts)
\t}
\t// [gallocr compute buffer reduction]
\t//
\t// From llama.cpp src/llama-context.cpp analysis:
\t//   The gallocr compute buffer on the PRIMARY CUDA device (CUDA0) scales with n_ctx
\t//   (total context length), NOT with n_ubatch. At n_ctx=262144 (num_parallel=2 × 131072,
\t//   32 attention heads, F32 KQ scores): buffer ≈ 11.6 GiB.
\t//   Tesla M10 has only 8 GiB → OOM. RTX3060 (12 GiB) has only 0.2 GiB margin → fragmentation.
\t//
\t//   NOTE: -b/-ub are added AFTER appendFlashAttentionArgs, so they are NOT in params
\t//   at this insertion point. We must modify -np and -c which ARE already in params here.
\t//
\t// Fix: reduce n_ctx and num_parallel by halving -c and setting -np 1.
\t//   -c 262144 → -c 131072: halves the KV cache and the gallocr compute buffer to ≈5.8 GiB
\t//   -np 2 → -np 1: single parallel slot (KV cache matches per-slot context length)
\t//   Tesla M10 (8 GiB) as CUDA0: 5.8 GiB compute + 0 model layers (greedy fills RTX first)
\t//   → 2.2 GiB free on M10 ✓
\t//
\t// selectGPUPool sets OLLAMA_MAX_BATCH_SIZE for the full pool path.
\tif _maxBatchStr := os.Getenv("OLLAMA_MAX_BATCH_SIZE"); _maxBatchStr != "" {
\t\tfor _pi := 0; _pi < len(params)-1; _pi++ {
\t\t\tswitch params[_pi] {
\t\t\tcase "-np", "--parallel":
\t\t\t\tif _cur, _e := strconv.Atoi(params[_pi+1]); _e == nil && _cur > 1 {
\t\t\t\t\tslog.Info("full pool: reducing num_parallel to halve gallocr compute buffer",
\t\t\t\t\t\t"from", _cur, "to", 1)
\t\t\t\t\tparams[_pi+1] = "1"
\t\t\t\t}
\t\t\tcase "-c", "--ctx-size":
\t\t\t\tif _cur, _e := strconv.Atoi(params[_pi+1]); _e == nil && _cur > 131072 {
\t\t\t\t\t_newCtx := _cur / 2
\t\t\t\t\tslog.Info("full pool: halving -c to match num_parallel=1",
\t\t\t\t\t\t"from", _cur, "to", _newCtx,
\t\t\t\t\t\t"compute_buffer_halved_to_gib", float64(_newCtx)*32*4*32768/(1<<30))
\t\t\t\t\tparams[_pi+1] = strconv.Itoa(_newCtx)
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\t'''
    if "selectGPUPool(&launch)" not in new_content:
        new_content = new_content.replace(
            INSERT_BEFORE,
            INLINE_BATCH_CAP + INSERT_BEFORE,
            1
        )

    path.write_text(new_content)
    print(f"  Dynamic GPU pool patch applied to {path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"Looking for {TARGET_FILE} under {root}...")
    target = find_target(root)
    if not target:
        print(f"  {TARGET_FILE} not found — skipping")
        sys.exit(0)
    print(f"  Target: {target}")
    if not patch(target):
        print("Dynamic pool patch failed — original behavior unchanged.", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
