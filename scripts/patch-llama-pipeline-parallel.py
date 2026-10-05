#!/usr/bin/env python3
"""
patch-llama-pipeline-parallel.py — adds LLAMA_PIPELINE_PARALLEL=0 as an opt-out for pipeline parallelism.

llama_context enables pipeline parallelism when several GPUs run a layer split, every layer is offloaded
(n_gpu_layers > n_layer_all) and no tensor overrides are active. The scheduler then keeps GGML_SCHED_MAX_COPIES (4)
copies of its input buffers, which multiplies the pinned host buffer (`CUDA_Host compute buffer`, the attention mask
n_ctx x n_ubatch x 2 B) by four and enlarges the per-GPU compute buffers (qwen3.6:35b on 4x Tesla M10, batch 64:
host buffer 32.8 -> 129.6 MiB, compute buffer 551 -> 648 MiB per GPU, no prefill gain). Where nothing may live in
host RAM, that is pure cost.

Without the variable nothing changes; with LLAMA_PIPELINE_PARALLEL=0 the condition is skipped and a log line says so.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [LLAMA_PIPELINE_PARALLEL patch]"
SOURCE_FILE = Path("src") / "llama-context.cpp"

OLD = '''        bool pipeline_parallel =
            model.n_devices() > 1 &&
'''
NEW = '''        ''' + PATCH_GUARD + ''' LLAMA_PIPELINE_PARALLEL=0 disables pipeline parallelism (saves host memory)
        const char * llama_pp_env = getenv("LLAMA_PIPELINE_PARALLEL");
        const bool llama_pp_allowed = !(llama_pp_env != nullptr && atoi(llama_pp_env) == 0);
        if (!llama_pp_allowed) {
            LLAMA_LOG_INFO("%s: pipeline parallelism disabled by LLAMA_PIPELINE_PARALLEL=0\\n", __func__);
        }
        bool pipeline_parallel =
            llama_pp_allowed &&
            model.n_devices() > 1 &&
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
    if content.count(OLD) != 1:
        print(f"  ERROR: pipeline_parallel condition not found exactly once in {path}", file=sys.stderr)
        return False
    path.write_text(content.replace(OLD, NEW, 1))
    print(f"  Pipeline-parallel opt-out applied to {path}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    paths = find_sources(Path(sys.argv[1]))
    if not paths:
        print("ERROR: src/llama-context.cpp not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(p) for p in paths) else 1


if __name__ == "__main__":
    sys.exit(main())
