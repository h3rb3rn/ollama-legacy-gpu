#!/usr/bin/env python3
"""
patch-llama-tier-fitting.py — Patches common_params_fit_impl in llama.cpp/common/fit.cpp
to support GPU-tier-aware layer distribution.

Context: llama.cpp fills GPUs back-to-front (highest CUDA index first). Combined with
CUDA_VISIBLE_DEVICES reordering (Tesla=low index, RTX=high index), RTX fills first.
BUT: the algorithm still distributes across ALL devices. This patch adds two-pass logic:
  Pass 1: fill "fast" GPUs (CUDA index >= OLLAMA_GPU_TIER_THRESHOLD) until done or full
  Pass 2: if layers remain, extend to "legacy" GPUs (CUDA index < threshold)

Result: models that fit in the RTX pool never touch Tesla GPUs.

Usage (inside Docker builder, after cmake configure):
    python3 scripts/patch-llama-tier-fitting.py <ollama-src-root>

Exits 0 on success or if file is not found (non-fatal).
Exits 1 only on structural errors that indicate the source changed significantly.

OLLAMA_GPU_TIER_THRESHOLD env var:
    0 (default)  = disabled, original behavior preserved
    N            = CUDA devices 0..N-1 are "legacy", N..nd-1 are "fast"

Override (Option 3 hybrid):
    When OLLAMA_TENSOR_SPLIT is set in the container env, Ollama passes --tensor-split
    to llama-server which bypasses common_params_fit_impl entirely.
    See scripts/patch-ollama-tensor-split.py.
"""

import sys
import re
import subprocess
from pathlib import Path

PATCH_GUARD    = "OLLAMA_GPU_TIER_THRESHOLD"
LOOP_MARKER    = "for (int id = nd - 1; id >= 0; id--) {"
DENSE_LOG      = 'filling dense layers back-to-front:'
SOURCE_FILE    = "common/fit.cpp"


def find_fit_cpp(ollama_root: Path) -> Path | None:
    """Find fit.cpp in the FetchContent'd llama.cpp source tree."""
    candidates = [
        ollama_root / "build" / "llama-server-cuda_v12" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "build" / "llama-server-cuda_v11" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "llama" / "llama.cpp" / SOURCE_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Fallback: search by content
    result = subprocess.run(
        ["grep", "-r", "-l", DENSE_LOG, str(ollama_root), "--include=*.cpp"],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.strip().splitlines():
        p = Path(line)
        if p.is_file():
            return p
    return None


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    if LOOP_MARKER not in content:
        print(f"  Loop marker not found in {path} — source may have changed", file=sys.stderr)
        return False

    lines = content.splitlines(keepends=True)

    # Find the back-to-front loop that fills dense layers
    loop_start = None
    for i, line in enumerate(lines):
        if LOOP_MARKER in line and i > 0 and (DENSE_LOG in "".join(lines[max(0,i-20):i])):
            loop_start = i
            break

    if loop_start is None:
        # Fallback: find first occurrence of LOOP_MARKER after DENSE_LOG
        dense_line = next((i for i, l in enumerate(lines) if DENSE_LOG in l), None)
        if dense_line is None:
            print(f"  Dense-log marker not found", file=sys.stderr)
            return False
        for i in range(dense_line, min(dense_line + 30, len(lines))):
            if LOOP_MARKER in lines[i]:
                loop_start = i
                break

    if loop_start is None:
        print(f"  Could not locate back-to-front loop in {path}", file=sys.stderr)
        return False

    # Find the end of the loop (track brace depth, starting from the opening brace)
    depth = 0
    loop_end = None
    for i in range(loop_start, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0 and i > loop_start:
            loop_end = i
            break

    if loop_end is None:
        print(f"  Could not find loop end brace in {path}", file=sys.stderr)
        return False

    print(f"  Loop found: lines {loop_start+1}–{loop_end+1} in {path.name}")

    # Extract the loop body (lines inside the for-loop, excluding the for() line itself)
    indent = re.match(r'^(\s*)', lines[loop_start]).group(1)
    body_lines = lines[loop_start + 1 : loop_end]  # lines inside the braces
    body_text = "".join(body_lines)

    # Build the replacement: extract body into lambda, then two-pass calls
    patch_code = (
        f"{indent}// --- ollama-legacy-gpu: GPU-tier-aware filling (patch-llama-tier-fitting.py) ---\n"
        f"{indent}// OLLAMA_GPU_TIER_THRESHOLD: CUDA devices >= threshold fill first (fast: RTX/Pascal+),\n"
        f"{indent}//   devices < threshold fill only if fast pool is exhausted (legacy: Maxwell/Kepler).\n"
        f"{indent}// Set to 0 (default) to keep original behavior.\n"
        f"{indent}// Overridden by OLLAMA_TENSOR_SPLIT (see patch-ollama-tensor-split.py).\n"
        f"{indent}{{\n"
        f"{indent}    const int _tier_threshold = []() -> int {{\n"
        f"{indent}        const char * _t = std::getenv(\"{PATCH_GUARD}\");\n"
        f"{indent}        return (_t && _t[0]) ? std::atoi(_t) : 0;\n"
        f"{indent}    }}();\n"
        f"{indent}    // Lambda capturing all loop-body variables by reference:\n"
        f"{indent}    auto _fill_devices = [&](int _from, int _to) {{\n"
        f"{indent}        for (int id = _from; id >= _to; id--) {{\n"
        f"{body_text}"
        f"{indent}        }}\n"
        f"{indent}    }};\n"
        f"{indent}    // Pass 1: fast GPUs (CUDA index >= threshold)\n"
        f"{indent}    _fill_devices((int)nd - 1, _tier_threshold);\n"
        f"{indent}    // Pass 2: legacy GPUs — only if layers remain\n"
        f"{indent}    if (_tier_threshold > 0) {{\n"
        f"{indent}        uint32_t _assigned = 0;\n"
        f"{indent}        for (size_t _jd = 0; _jd < nd; _jd++) {{ _assigned += ngl_per_device[_jd].n_layer; }}\n"
        f"{indent}        if (_assigned < hp_ngl + 1) {{\n"
        f"{indent}            LOG_TRC(\"%s: fast GPUs filled, extending to legacy GPUs \"\n"
        f"{indent}                    \"(threshold=%d, remaining=%u)\\n\",\n"
        f"{indent}                    __func__, _tier_threshold, hp_ngl + 1 - _assigned);\n"
        f"{indent}            _fill_devices(_tier_threshold - 1, 0);\n"
        f"{indent}        }} else {{\n"
        f"{indent}            LOG_TRC(\"%s: all layers fit in fast GPUs, legacy GPUs idle (threshold=%d)\\n\",\n"
        f"{indent}                    __func__, _tier_threshold);\n"
        f"{indent}        }}\n"
        f"{indent}    }}\n"
        f"{indent}}} // end GPU-tier-aware filling\n"
    )

    # Replace the original loop
    original_block = "".join(lines[loop_start : loop_end + 1])
    new_content = content.replace(original_block, patch_code, 1)

    if new_content == content:
        print(f"  Replacement had no effect — source structure mismatch", file=sys.stderr)
        return False

    path.write_text(new_content)
    print(f"  GPU-tier-fitting patch applied to {path}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"Looking for llama.cpp {SOURCE_FILE} under {root}...")

    target = find_fit_cpp(root)
    if not target:
        print(f"  {SOURCE_FILE} not found — skipping (cmake configure may not have run yet)")
        sys.exit(0)

    if not patch(target):
        print("Patch failed — build continues with original fitting behavior.", file=sys.stderr)
        sys.exit(0)  # non-fatal


if __name__ == "__main__":
    main()
