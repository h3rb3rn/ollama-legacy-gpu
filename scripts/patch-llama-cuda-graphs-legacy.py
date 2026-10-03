#!/usr/bin/env python3
"""
patch-llama-cuda-graphs-legacy.py — lets ggml-cuda use CUDA graphs on pre-Volta GPUs.

ggml_cuda_graph_set_enabled() hard-disables CUDA graphs for CC < 7.0
("disabling CUDA graphs due to GPU architecture"). On Maxwell every token then
launches the whole graph kernel by kernel from the host. For qwen35moe (hybrid SSM +
MoE, ~3800 graph nodes) the llama-server main thread sits at 100 % of a slow CPU core
while the GPUs idle at ~20 % (RTX-TESLA-GAPS-PROMPT.md, Gap 3; N11-M10: i5-3470T,
6.4 tok/s).  CUDA graphs themselves exist on every CC >= 3.0.

The patch adds a runtime switch, GGML_CUDA_GRAPHS_LEGACY=1, so one image can be A/B
tested; without the variable behaviour is unchanged.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [GGML_CUDA_GRAPHS_LEGACY patch]"
SOURCE_FILE = Path("ggml") / "src" / "ggml-cuda" / "ggml-cuda.cu"

OLD = "        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {\n"
NEW = "        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA && !ggml_cuda_legacy_graphs_enabled()) {\n"
FUNC = "static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {\n"
HELPER = PATCH_GUARD + ''' opt-in CUDA graphs on CC < 7.0
static bool ggml_cuda_legacy_graphs_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_GRAPHS_LEGACY");
        return env != nullptr && atoi(env) == 1;
    }();
    return enabled;
}

'''


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
    if content.count(OLD) != 1 or content.count(FUNC) != 1:
        print(f"  ERROR: CUDA graph arch check not found exactly once in {path}", file=sys.stderr)
        return False
    content = content.replace(FUNC, HELPER + FUNC, 1).replace(OLD, NEW, 1)
    path.write_text(content)
    print(f"  Legacy CUDA graph switch applied to {path}")
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
