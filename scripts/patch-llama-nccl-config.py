#!/usr/bin/env python3
"""
patch-llama-nccl-config.py — lets llama.cpp build against NCCL older than 2.17.

llama.cpp b12111 (Ollama v0.40.15) sets `ncclConfig_t::maxCTAs` for the NVIDIA DGX Spark, so it does not take 64
channels there. That member exists from NCCL 2.17 on. Ubuntu 22.04 (the base of the cuda12-maxwell and cuda11-legacy
images) ships NCCL 2.16.5, and nvcc stops with `class "ncclConfig_v21400" has no member "maxCTAs"`.

The patch compiles the DGX Spark setting only when the NCCL headers have the member (NCCL_VERSION_CODE >= 2.17.1).
The setting is irrelevant for the GPUs of these images (CC 3.7 to 9.0); with newer NCCL nothing changes.
"""

import re
import sys
from pathlib import Path

PATCH_GUARD = "// [NCCL config patch]"
SOURCE_FILE = Path("ggml") / "src" / "ggml-cuda" / "ggml-cuda.cu"

# llama.cpp renamed GGML_CUDA_CC_DGX_SPARK to GGML_CUDA_CC_GB10 (b12651, Ollama v0.40.18); accept both.
OLD_RE = re.compile(
    r"(?P<indent>[ \t]*)if \(GGML_CUDA_CC_IS_NVIDIA\(cc\) && cc == (?P<const>GGML_CUDA_CC_(?:DGX_SPARK|GB10))\) \{\n"
    r"[ \t]*config\.maxCTAs = 8;\n"
    r"[ \t]*\}\n"
)


def build_new(m: re.Match) -> str:
    i = m.group("indent")
    return (
        f"{i}{PATCH_GUARD} ncclConfig_t::maxCTAs exists from NCCL 2.17.1 on; older headers (Ubuntu 22.04: 2.16.5) do not compile it\n"
        f"#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 17, 1)\n"
        f"{i}if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == {m.group('const')}) {{\n"
        f"{i}    config.maxCTAs = 8;\n"
        f"{i}}}\n"
        f"#else\n"
        f"{i}GGML_UNUSED(cc);\n"
        f"#endif\n"
    )


def find_sources(root: Path):
    out = []
    for variant in ("cuda_v12", "cuda_v13", "cuda_v11"):
        base = root / "build" / f"llama-server-{variant}" / "_deps" / "llama_cpp-src"
        if (base / SOURCE_FILE).is_file():
            out.append(base / SOURCE_FILE)
    return out


def patch(path: Path) -> bool:
    content = path.read_text()
    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True
    if len(OLD_RE.findall(content)) != 1:
        print(f"  ERROR: DGX Spark maxCTAs block not found exactly once in {path}", file=sys.stderr)
        return False
    path.write_text(OLD_RE.sub(build_new, content, count=1))
    print(f"  NCCL maxCTAs guard applied to {path}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    paths = find_sources(Path(sys.argv[1]))
    if not paths:
        print("ERROR: ggml-cuda.cu not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(p) for p in paths) else 1


if __name__ == "__main__":
    sys.exit(main())
