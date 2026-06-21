#!/usr/bin/env python3
"""
patch-ollama-fa.py — Patches llm/llama_server.go to enable Flash Attention
for the fast GPU pool when OLLAMA_GPU_TIER_THRESHOLD is set.

Context:
  Ollama's LlamaServerFlashAttention() calls ml.FlashAttentionSupported(gpus)
  which returns false if ANY visible GPU lacks FA support (CC < 7.0).
  With CUDA_VISIBLE_DEVICES reordering, legacy GPUs (Tesla M10/M60, GTX 1060)
  are at low CUDA indices (CUDA0..threshold-1) and fast GPUs (RTX) at high
  indices (CUDA threshold..nd-1).

  When the tier-fitting patch places all model layers on fast GPUs only,
  legacy GPUs are idle — but Ollama still disables FA globally because they
  appear in the visible device list.

This patch adds tier-aware FA to LlamaServerFlashAttention():
  If OLLAMA_GPU_TIER_THRESHOLD > 0:
    → Only check FA support on fast GPUs (gpus[threshold:])
    → If all fast GPUs support FA (CC >= 7.0, driver >= 7): enable FA
  If OLLAMA_GPU_TIER_THRESHOLD == 0 (default):
    → Original behavior: check all visible GPUs

Result:
  qwen3.6:35b with TIER_THRESHOLD=7 → only RTX2060/3060 get layers
  → all fast GPUs have CC 7.5/8.6 → FA enabled → ~50-60 tok/s

Usage (after Go source is available, before go build):
    python3 scripts/patch-ollama-fa.py <ollama-source-root>

OLLAMA_GPU_TIER_THRESHOLD env var must match the value used in gpu-detect.sh.

This patch is compatible with Ollama's upstream API — it does not change any
exported function signatures or behavior when TIER_THRESHOLD is 0.
"""

import sys
import re
from pathlib import Path

PATCH_GUARD  = "OLLAMA_GPU_TIER_THRESHOLD"
TARGET_FILE  = "llm/llama_server.go"
FUNC_MARKER  = "func LlamaServerFlashAttention(gpus []ml.DeviceInfo) ml.FlashAttentionType {"
OLD_LAST_CHECK = "if !ml.FlashAttentionSupported(gpus) {\n\t\treturn ml.FlashAttentionDisabled\n\t}\n\treturn ml.FlashAttentionAuto"


def find_target(ollama_root: Path) -> Path | None:
    candidates = [
        ollama_root / TARGET_FILE,
        ollama_root / "src" / TARGET_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    if FUNC_MARKER not in content:
        print(f"  Function marker not found in {path}", file=sys.stderr)
        return False

    if OLD_LAST_CHECK not in content:
        # Try with tab variations
        alt = "if !ml.FlashAttentionSupported(gpus) {\n\t\treturn ml.FlashAttentionDisabled\n\t}\n\treturn ml.FlashAttentionAuto"
        alt2 = "if !ml.FlashAttentionSupported(gpus) {\n        return ml.FlashAttentionDisabled\n    }\n    return ml.FlashAttentionAuto"
        for candidate in [alt, alt2]:
            if candidate in content:
                old = candidate
                break
        else:
            print(f"  Could not locate FA support check in {path}", file=sys.stderr)
            # Try regex
            m = re.search(
                r'if !ml\.FlashAttentionSupported\(gpus\) \{\s*\n\s*return ml\.FlashAttentionDisabled\s*\n\s*\}\s*\n\s*return ml\.FlashAttentionAuto',
                content
            )
            if not m:
                return False
            old = m.group(0)
    else:
        old = OLD_LAST_CHECK

    # Determine indentation from the existing return statement
    # Find the line with "return ml.FlashAttentionAuto"
    lines = content.splitlines()
    indent = "\t"
    for line in lines:
        if "return ml.FlashAttentionAuto" in line:
            indent = re.match(r'^(\s*)', line).group(1)
            break

    new = f"""\
// [OLLAMA_GPU_TIER_THRESHOLD patch] When tier threshold is set, only check
{indent}// Flash Attention support on fast GPUs (CUDA index >= threshold).
{indent}// Legacy GPUs at low CUDA indices (Tesla/GTX) may be idle for small models
{indent}// and should not block FA on the fast RTX pool.
{indent}// When threshold == 0 (default): original behavior, check all visible GPUs.
{indent}gpusForFA := gpus
{indent}if t, err := func() (int, error) {{
{indent}\tv := os.Getenv("{PATCH_GUARD}")
{indent}\tif v == "" {{
{indent}\t\treturn 0, fmt.Errorf("not set")
{indent}\t}}
{indent}\treturn strconv.Atoi(v)
{indent}}}(); err == nil && t > 0 && t < len(gpus) {{
{indent}\tgpusForFA = gpus[t:] // only fast GPUs (CUDA index >= threshold)
{indent}\tif len(gpusForFA) == 0 {{
{indent}\t\tgpusForFA = gpus // safety fallback
{indent}\t}}
{indent}}}
{indent}if !ml.FlashAttentionSupported(gpusForFA) {{
{indent}\treturn ml.FlashAttentionDisabled
{indent}}}
{indent}return ml.FlashAttentionAuto"""

    new_content = content.replace(old, new, 1)
    if new_content == content:
        print(f"  Replacement had no effect", file=sys.stderr)
        return False

    path.write_text(new_content)
    print(f"  Tier-aware FA patch applied to {path}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"Looking for {TARGET_FILE} under {root}...")

    target = find_target(root)
    if not target:
        print(f"  {TARGET_FILE} not found — skipping (non-fatal)")
        sys.exit(0)

    print(f"  Target: {target}")

    if not patch(target):
        print("FA patch failed — building with original FA behavior.", file=sys.stderr)
    sys.exit(0)  # non-fatal: build continues


if __name__ == "__main__":
    main()
