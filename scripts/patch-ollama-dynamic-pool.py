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
    → Use all visible GPUs (no CUDA_VISIBLE_DEVICES override)
    → Keep FA setting from environment (OFF when Tesla present)
    → Large models (>60 GB): full 113 GB pool for llama4:scout etc.

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
// Note: "strings" and "os" are already imported in llama_server.go.

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
	} else {
		// Model requires full GPU pool (Tesla present).
		// Do NOT force greedy fill AND disable tier-fitting patches.
		//
		// Why tier-patches must be disabled for full pool:
		//   The tier-fitting patch (Patch 2) assigns layers to Tesla M10 in its
		//   second pass when RTX doesn't hold all layers. Tesla M10 (8 GB) then
		//   needs a ~12 GiB compute buffer for non-FA attention (Q×K^T at 131072+
		//   context × batch=512) — exceeding its 8 GB → OOM at context init.
		//
		//   Standard Ollama fitting correctly computes the compute buffer estimate
		//   per GPU in margins_s[]. If Tesla M10's compute budget is negative
		//   (12 GB compute > 8 GB VRAM), it gets 0 layers and 0 compute allocation.
		//   This is what makes standard Ollama work with llama4:scout on this hardware.
		//
		//   Setting OLLAMA_GPU_TIER_THRESHOLD=0 disables our tier-filling patches
		//   for this model load, letting the standard algorithm run unmodified.
		// Full pool: all 12 GPUs, FA off (Tesla/GTX lack FA support).
		// Use greedy fill (FORCE_GPU_LAYERS=999) with reduced batch size.
		//
		// Without Flash Attention the prefill compute buffer per GPU is:
		//   batch_size × context × num_heads × 4 bytes
		//   = 512 × 131072 × 32 × 4 ≈ 11.6 GiB  (at default batch=512)
		// This exceeds Tesla M10 (8 GiB) and M60 (6.7 GiB available), making
		// those GPUs unusable and forcing model layers to CPU.
		//
		// With batch=64: 64 × 131072 × 32 × 4 ≈ 1.07 GiB per GPU.
		// Every GPU in the pool can hold this, so the greedy fill assigns layers
		// to RTX3060 → RTX2060 → GTX1060 → M60 → M10, stopping as soon as all
		// layers are placed (e.g. 9 GPUs for llama4:scout, not all 12).
		//
		// Keep original CUDA order (worst→best: M10=CUDA0 … RTX3060=CUDA11).
		// Greedy fill runs from CUDA11 (RTX3060, best bandwidth) down to CUDA0,
		// so it fills RTX GPUs first and only extends to Tesla if needed.
		// With batch=64 the primary orchestrator (CUDA0=M10, 8 GiB) holds only
		// 1.07 GiB compute buffer — well within its VRAM budget.
		// DO NOT reverse CUDA order: that would fill M10 first (wrong direction).
		slog.Info("dynamic GPU pool: model exceeds fast pool, using all GPUs with capped batch (greedy fill)",
			"model_gb", modelBytes/(1<<30),
			"threshold_gb", thresholdBytes/(1<<30))
		launch.extraEnvs["OLLAMA_FORCE_GPU_LAYERS"] = "999"
		launch.extraEnvs["OLLAMA_MAX_BATCH_SIZE"] = "64"
		os.Setenv("OLLAMA_FORCE_GPU_LAYERS", "999")
		os.Setenv("OLLAMA_MAX_BATCH_SIZE", "64")
		os.Setenv("OLLAMA_GPU_TIER_THRESHOLD", "0")
		// FA off: full pool includes Tesla/GTX which lack FA support
		os.Setenv("OLLAMA_FLASH_ATTENTION", "false")
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
\t// [OLLAMA_MAX_BATCH_SIZE patch] Cap ONLY --ubatch-size / -ub in params.
\t//
\t// From llama.cpp analysis (src/llama-context.cpp:205):
\t//   n_ubatch (--ubatch-size) determines the compute buffer size.
\t//   n_batch  (--batch-size)  is the logical batch, split into n_ubatch chunks.
\t//   compute buffer = n_ubatch × ctx × n_heads × head_dim × 4 bytes per GPU
\t//   At n_ubatch=512, ctx=131072, 32 heads: 8 GiB per attention layer → OOM on M10 (8 GiB)
\t//   At n_ubatch=64:  64 × 131072 × 32 × 4 = 1.07 GiB → fits on all GPUs
\t//
\t// We cap only n_ubatch (not n_batch) so prefill efficiency is preserved:
\t//   n_batch=512, n_ubatch=64: 512 logical tokens per step, chunked into 8 × 64-token passes
\t//   n_batch=64,  n_ubatch=64: 64 logical tokens per step (8x more llama_decode calls)
\t//
\t// selectGPUPool sets OLLAMA_MAX_BATCH_SIZE for the full pool path (Tesla present).
\tif _maxBatchStr := os.Getenv("OLLAMA_MAX_BATCH_SIZE"); _maxBatchStr != "" {
\t\tif _maxBatch, _bErr := strconv.Atoi(_maxBatchStr); _bErr == nil && _maxBatch > 0 {
\t\t\t// Only cap the physical micro-batch (--ubatch-size / -ub), not the logical batch
\t\t\t_ubatchFlags := map[string]bool{"--ubatch-size": true, "-ub": true}
\t\t\tfor _bi := 0; _bi < len(params)-1; _bi++ {
\t\t\t\tif _ubatchFlags[params[_bi]] {
\t\t\t\t\tif _cur, _e := strconv.Atoi(params[_bi+1]); _e == nil && _cur > _maxBatch {
\t\t\t\t\t\tslog.Info("ubatch size capped for multi-GPU pool (reduces compute buffer)",
\t\t\t\t\t\t\t"from", _cur, "to", _maxBatch,
\t\t\t\t\t\t\t"compute_buffer_gb", float64(_maxBatch)*262144*32*4/(1<<30))
\t\t\t\t\t\tparams[_bi+1] = strconv.Itoa(_maxBatch)
\t\t\t\t\t}
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
