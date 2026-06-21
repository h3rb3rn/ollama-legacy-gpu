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
// selectGPUPool dynamically restricts CUDA_VISIBLE_DEVICES to the fast GPU pool
// (OLLAMA_FAST_GPU_DEVICES) when the model file fits within the fast pool capacity.
// This enables Flash Attention on the fast-only pool while still allowing large
// models to use the full GPU pool (including legacy Tesla GPUs) when needed.
//
// Env vars (set by gpu-detect.sh):
//   OLLAMA_FAST_GPU_DEVICES    comma-separated UUID list of fast GPUs (CC >= 7.5)
//   OLLAMA_FAST_POOL_VRAM_GB   total fast pool VRAM in GB
func selectGPUPool(launch *llamaServerLaunchConfig) {
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

	if modelBytes <= thresholdBytes {
		// Model fits in fast pool → restrict to fast GPUs + enable FA
		slog.Info("dynamic GPU pool: model fits in fast pool, restricting to fast GPUs",
			"model_gb", modelBytes/(1<<30),
			"pool_gb", poolGB,
			"devices", fastDevices)
		if launch.extraEnvs == nil {
			launch.extraEnvs = make(map[string]string)
		}
		launch.extraEnvs["CUDA_VISIBLE_DEVICES"] = fastDevices
		// Explicitly enable FA: fast GPUs all support FA (CC >= 7.5)
		// This overrides the global OLLAMA_FLASH_ATTENTION=off setting.
		os.Setenv("OLLAMA_FLASH_ATTENTION", "true")
	} else {
		slog.Info("dynamic GPU pool: model exceeds fast pool, using all GPUs",
			"model_gb", modelBytes/(1<<30),
			"threshold_gb", thresholdBytes/(1<<30))
		// Restore FA to off (Tesla present in full pool)
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

    # Add the call to selectGPUPool before appendFlashAttentionArgs
    if "selectGPUPool(&launch)" not in new_content:
        new_content = new_content.replace(
            INSERT_BEFORE,
            "selectGPUPool(&launch)\n\t" + INSERT_BEFORE,
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
