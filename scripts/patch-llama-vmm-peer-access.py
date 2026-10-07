#!/usr/bin/env python3
"""
patch-llama-vmm-peer-access.py — keeps the CUDA VMM pool usable with more than nine GPUs in one process.

ggml_cuda_pool_vmm::alloc grants cuMemSetAccess to every peer-accessible device whenever `use_peer_access` is true.
In builds with GGML_USE_NCCL (all our images ship libnccl) the flag is forced to true. CUDA allows at most eight peer
mappings per allocation, so with more than nine visible GPUs the first pool allocation fails with
`CUDA error: peer mapping resources exhausted` (CUDA_ERROR_TOO_MANY_PEERS) and the llama-server aborts
(12x Tesla M60 in one instance: fails with 12 visible GPUs, works with 8 and 9 unpatched).

The patch forces peer access in NCCL builds only up to eight devices (conservative: nine devices also work unpatched). Beyond that the pool grants access to the owning
device only (the layer split does not read pool memory across devices). An explicit GGML_CUDA_P2P keeps its meaning.
Up to eight GPUs nothing changes.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [VMM peer access patch]"
SOURCE_FILE = Path("ggml") / "src" / "ggml-cuda" / "ggml-cuda.cu"

OLD = '''#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)
'''
NEW = '''#if defined(GGML_USE_NCCL)
            ''' + PATCH_GUARD + ''' CUDA allows at most 8 peer mappings per allocation: force peer access only up to 8 devices
            if (ggml_cuda_info().device_count <= 8) {
                use_peer_access = true;
            }
#endif // defined(GGML_USE_NCCL)
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
        print(f"  ERROR: NCCL peer-access block not found exactly once in {path}", file=sys.stderr)
        return False
    path.write_text(content.replace(OLD, NEW, 1))
    print(f"  VMM peer-access limit applied to {path}")
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
