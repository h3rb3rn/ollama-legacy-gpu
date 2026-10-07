#!/usr/bin/env python3
"""
patch-llama-kolibri1.py — adds the `kolibri1` architecture (Aleph Alpha Kolibri-1, 78B MoE) to llama.cpp.

Upstream llama.cpp (b11232 up to b11351) answers `unknown model architecture: 'kolibri1'`. The runtime part (src/) of the patch
published with Hob-forge/Kolibri-1-GGUF (MIT, author Seraphiel102) is kept in patches/kolibri1/kolibri1-llama.cpp.patch:
new gating mode SIGMOID_LOGIT_ADD (select top-k on logits + expert bias, weight by the unbiased sigmoid), the model graph
src/models/kolibri1.cpp (sliding-window 4:1 pattern, sandwich norms, shared expert) and the `kolibri1` tokenizer type.
Converter, gguf-py and test hunks are not needed to load an existing GGUF and are left out.

The patch is applied with `patch -p1` after a dry run; it fails closed if any hunk does not match.
src/CMakeLists.txt collects models/*.cpp with a GLOB at configure time, and the patch runs after the configure step:
the script touches src/CMakeLists.txt so that `cmake --build` reconfigures and compiles the new file.
"""

import os
import subprocess
import sys
from pathlib import Path

PATCH_GUARD = "LLM_ARCH_KOLIBRI1"
GUARD_FILE = Path("src") / "llama-arch.h"
PATCH_FILE = Path(__file__).resolve().parent.parent / "patches" / "kolibri1" / "kolibri1-llama.cpp.patch"


def find_sources(root: Path):
    out = []
    for variant in ("cuda_v12", "cuda_v13", "cuda_v11"):
        base = root / "build" / f"llama-server-{variant}" / "_deps" / "llama_cpp-src"
        if (base / GUARD_FILE).is_file():
            out.append(base)
    return out


def run_patch(base: Path, patch_file: Path, dry: bool) -> bool:
    cmd = ["patch", "-p1", "--forward", "--batch", "-i", str(patch_file)]
    if dry:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=base, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout + result.stderr, file=sys.stderr)
    return result.returncode == 0


def patch(base: Path, patch_file: Path = PATCH_FILE) -> bool:
    if PATCH_GUARD in (base / GUARD_FILE).read_text():
        print(f"  Already patched: {base}")
        return True
    if not patch_file.is_file():
        print(f"  ERROR: {patch_file} missing", file=sys.stderr)
        return False
    if not run_patch(base, patch_file, dry=True):
        print(f"  ERROR: kolibri1 patch does not apply to {base}", file=sys.stderr)
        return False
    if not run_patch(base, patch_file, dry=False):
        return False
    os.utime(base / "src" / "CMakeLists.txt")  # re-run the models/*.cpp GLOB
    print(f"  kolibri1 architecture applied to {base}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    bases = find_sources(Path(sys.argv[1]))
    if not bases:
        print("ERROR: llama.cpp source not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(b) for b in bases) else 1


if __name__ == "__main__":
    sys.exit(main())
