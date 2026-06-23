#!/usr/bin/env python3
"""
patch-ollama-batch.py — Patches llm/llama_server.go to support OLLAMA_MAX_BATCH_SIZE.

Without Flash Attention, the non-FA attention compute buffer for one layer is:
  batch_size × context_length × num_heads × head_dim × bytes_per_element
  = 512 × 131072 × 32 × 128 × 4 ≈ 11.6 GiB per GPU

This exceeds the VRAM of Tesla M10 (8 GiB) and M60 (6.7 GiB available), which
prevents these GPUs from computing attention layers and forces most model layers
onto CPU despite having 114 GiB total VRAM across 12 GPUs.

With batch_size = 64:
  64 × 131072 × 32 × 128 × 4 ≈ 1.07 GiB per GPU

This fits on all GPUs in the pool including Tesla M10, allowing the greedy fill
to distribute llama4:scout (49 layers) across RTX→GTX→M60→M10 in order of
bandwidth, using only as many GPUs as needed and stopping when all layers fit.

Usage:
  Set OLLAMA_MAX_BATCH_SIZE=64 in the environment (done by selectGPUPool for
  the full pool path). This patch makes Ollama read and apply the cap.

  Build: applied before `go build` inside the Docker builder stage.
"""

import sys
import re
from pathlib import Path

PATCH_GUARD = "OLLAMA_MAX_BATCH_SIZE"
TARGET_FILE = "llm/llama_server.go"

# The batch size limiting code to insert.
# We look for the line that appends --batch-size to the params slice and
# insert a cap before it.  In Ollama the pattern is typically:
#   params = append(params, "--batch-size", fmt.Sprintf("%d", numBatch))
# We locate the numBatch assignment and add a cap after it.
BATCH_CAP_CODE = '''
	// [OLLAMA_MAX_BATCH_SIZE patch] Cap batch size for full-pool (Tesla present) inference.
	// Without Flash Attention, the non-FA attention compute buffer per GPU is
	//   batch_size × context_length × num_heads × 4 bytes
	// For llama4:scout (32 heads, ctx=131072): 512 × 131072 × 32 × 4 ≈ 11.6 GiB.
	// Tesla M10 (8 GiB) and M60 (6.7 GiB) cannot hold this, excluding them from
	// the compute pipeline and pushing model layers to CPU.
	// With batch_size = 64: ≈ 1.07 GiB — fits on every GPU in the pool.
	// selectGPUPool() sets OLLAMA_MAX_BATCH_SIZE=64 for the full pool path.
	if maxBatchStr := os.Getenv("OLLAMA_MAX_BATCH_SIZE"); maxBatchStr != "" {
		if maxBatch, err := strconv.Atoi(maxBatchStr); err == nil && maxBatch > 0 {
			if numBatch > maxBatch {
				slog.Info("batch size capped for multi-GPU pool",
					"original", numBatch, "capped", maxBatch,
					"reason", "OLLAMA_MAX_BATCH_SIZE")
				numBatch = maxBatch
			}
			if numUBatch > maxBatch {
				numUBatch = maxBatch
			}
		}
	}
'''

# Marker: search for the line that appends --batch-size to find insertion point
SEARCH_PATTERN = re.compile(
    r'(params\s*=\s*append\s*\(params,\s*"--batch-size")',
    re.MULTILINE
)

# Alternative: look for numUBatch assignment as a simpler anchor
SEARCH_PATTERN_ALT = re.compile(
    r'(numUBatch\s*:?=\s*numBatch)',
    re.MULTILINE
)


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

    # Check that strconv is imported (needed for Atoi)
    if '"strconv"' not in content:
        # Add strconv import if missing
        content = content.replace(
            '"strings"',
            '"strconv"\n\t"strings"',
            1
        )
        print("  Added strconv import")

    # Try primary pattern: insert before --batch-size append
    m = SEARCH_PATTERN.search(content)
    if m:
        insert_pos = m.start()
        # Find start of the line
        line_start = content.rfind('\n', 0, insert_pos) + 1
        indent = re.match(r'\s*', content[line_start:]).group(0)
        new_content = content[:line_start] + BATCH_CAP_CODE + content[line_start:]
        path.write_text(new_content)
        print(f"  Batch cap patch applied (before --batch-size append) in {path.name}")
        return True

    # Try alt pattern: insert after numUBatch := numBatch
    m = SEARCH_PATTERN_ALT.search(content)
    if m:
        # Insert after the matched line
        end_of_line = content.find('\n', m.end()) + 1
        new_content = content[:end_of_line] + BATCH_CAP_CODE + content[end_of_line:]
        path.write_text(new_content)
        print(f"  Batch cap patch applied (after numUBatch assignment) in {path.name}")
        return True

    print(f"  WARNING: could not find anchor in {path.name}; patch skipped", file=sys.stderr)
    print("  Searched for: '--batch-size' append or 'numUBatch := numBatch'", file=sys.stderr)
    return False


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
    patch(target)


if __name__ == "__main__":
    main()
