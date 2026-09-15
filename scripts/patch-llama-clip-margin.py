#!/usr/bin/env python3
"""
patch-llama-clip-margin.py — Reserves a CLIP/vision compute-buffer margin
before common/common.cpp's common_init_result constructor calls
common_fit_params() to decide GPU layer placement.

Root cause (see BUG-hybrid-arch-degeneration.md, Finding 1, in the parent
ollama-legacy-gpu repo): common_get_device_memory_data_impl() (fit.cpp) only
ever constructs a text-only llama_context to probe per-device memory, so the
fitting probe has zero visibility into the vision/CLIP tower's compute
buffer, which mtmd reserves separately, unconditionally, and independent of
context length once it initializes.

On small-VRAM legacy GPUs (Tesla M10, 8 GiB) this buffer alone can exceed the
probe's entire estimated surplus: the probe sees e.g. ~3.3 GiB needed for a
full 34-layer model (comfortably under 8 GiB) and commits to offloading it,
but the real reservation once CLIP is accounted for is closer to 8 GiB
total, so the fit collapses to near-zero GPU layers at actual allocation
time — the model then runs almost entirely on CPU.

Fix: when params.mmproj is set (and mmproj isn't disabled/CPU-only), inflate
fit_params_target[0] by a fixed ~4.75 GiB margin before calling
common_fit_params(). Padded from two observed real CLIP compute-buffer
reservations (4628 MiB, 4657 MiB), consistently on device 0 regardless of
--main-gpu. This is a fast, low-risk first step, not a computed exact value
— a follow-up could query mtmd's own compute-buffer estimate directly for a
tighter fit.

Usage (inside Docker builder, after cmake configure, before cmake build):
    python3 scripts/patch-llama-clip-margin.py <ollama-src-root>
"""

import re
import sys
import subprocess
from pathlib import Path

PATCH_GUARD = "// [OLLAMA_CLIP_MARGIN_v2_DIAG]"
SOURCE_FILE = "common/common.cpp"

# Anchor on just the unique call-start substring, not the preceding LOG_INF
# lines (their exact wrapping/formatting can differ across llama.cpp
# versions/forks — this call signature is far more stable).
ANCHOR = "common_fit_params(params.model.path.c_str(), &mparams, &cparams,"


def find_common_cpp(ollama_root: Path) -> Path | None:
    candidates = [
        ollama_root / "build" / "llama-server-cuda_v12" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "build" / "llama-server-cuda_v11" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "llama" / "llama.cpp" / SOURCE_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return c
    result = subprocess.run(
        ["grep", "-r", "-l", "common_fit_params(params.model.path.c_str()", str(ollama_root), "--include=*.cpp"],
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

    idx = content.find(ANCHOR)
    if idx == -1:
        print(f"  Anchor not found in {path} — layout may have changed upstream", file=sys.stderr)
        return False

    # Indent = whitespace from the start of the anchor's line.
    line_start = content.rfind("\n", 0, idx) + 1
    indent = content[line_start:idx]
    if not re.fullmatch(r"[ \t]*", indent):
        # Anchor wasn't at the start of its line (unexpected) — bail out safely.
        print(f"  Anchor found but not at line start in {path} — refusing to guess indent", file=sys.stderr)
        return False

    # DIAGNOSTIC BUILD: log the pre-existing margin/free values without adding
    # anything, to determine ground truth before tuning the actual margin.
    injected = (
        f"{indent}{PATCH_GUARD}\n"
        f"{indent}if (!params.mmproj.path.empty()) {{\n"
        f'{indent}    LOG_INF("%s: [DIAG] mmproj set, no_mmproj=%d, mmproj_use_gpu=%d, '
        f'fit_params_target[0]=%zu MiB (BEFORE any patch addition)\\n",\n'
        f"{indent}            __func__, (int)params.no_mmproj, (int)params.mmproj_use_gpu,\n"
        f"{indent}            params.fit_params_target.empty() ? (size_t)0 : params.fit_params_target[0] / (1024 * 1024));\n"
        f"{indent}}}\n"
    )
    content = content[:line_start] + injected + content[line_start:]
    path.write_text(content)
    print(f"  CLIP compute-buffer margin patch applied to {path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    ollama_root = Path(sys.argv[1])
    print(f"Looking for llama.cpp {SOURCE_FILE} under {ollama_root}...")
    target = find_common_cpp(ollama_root)
    if not target:
        print(f"  {SOURCE_FILE} not found — skipping (cmake configure may not have run yet)")
        sys.exit(0)  # non-fatal

    print(f"  Target: {target}")
    if not patch(target):
        print("Patch failed — build continues with original behavior.", file=sys.stderr)

    # Always exit 0 (non-fatal)
    sys.exit(0)


if __name__ == "__main__":
    main()
