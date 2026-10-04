#!/usr/bin/env python3
"""
patch-llama-input-gpu.py — keeps the input layer (token embedding table) in VRAM instead of
host memory (operator requirement: nothing but the unavoidable staging buffers on the CPU).

Upstream llama_model::load_tensors always pins the input layer to the CPU:

    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

For qwen35moe (vocab 248320 x 2048, Q4_K) that is a 272.81 MiB buffer in host memory
(`CUDA_Host` / `CPU_Mapped model buffer`) that never reaches the GPU. With this patch the input
layer follows the device of layer 0 (get_layer_buft_list(0)), i.e. it lands in the same VRAM
as the first trunk layer; if layer 0 itself is not offloaded the helper returns the CPU entry,
so the behaviour degrades exactly like upstream.

The placement still goes through llama_model::create_tensor's buffer-type selection
(weight_buft_supported / select_weight_buft), which asks the device whether it can run
GET_ROWS on the embedding type and walks on to the next buffer type (ending at the CPU) if not.
That is why this is done in the loader and not with `--override-tensor token_embd.weight=CUDA0`:
an override skips that check and aborts the scheduler ("pre-allocated tensor (token_embd.weight)
in a buffer (CUDA0) that cannot run the operation", observed on N11-M10, llama.cpp b11232).

Opt out with LLAMA_INPUT_LAYER_GPU=0 (restores the upstream CPU placement).
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [INPUT_LAYER_GPU patch]"
SOURCE_FILE = Path("src") / "llama-model.cpp"

INCLUDE_OLD = "#include <cstring>\n"
INCLUDE_NEW = "#include <cstdlib>\n#include <cstring>\n"

OLD = '''    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
'''
NEW = '''    // assign the input layer
    ''' + PATCH_GUARD + ''' upstream keeps the input layer (token embedding) on the CPU because
    // there is "very little benefit to offloading" it. This fork keeps all weights in VRAM: the input
    // layer follows layer 0 (falls back to the CPU entry when layer 0 is not offloaded, and
    // create_tensor falls back along the buffer list when the device cannot run GET_ROWS on it).
    // LLAMA_INPUT_LAYER_GPU=0 restores the upstream placement.
    {
        const char * env_input_gpu = std::getenv("LLAMA_INPUT_LAYER_GPU");
        if (env_input_gpu != nullptr && std::strcmp(env_input_gpu, "0") == 0) {
            pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
        } else {
            pimpl->dev_input = get_layer_buft_list(0);
        }
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
    if content.count(OLD) != 1:
        print(f"  ERROR: input-layer block not found exactly once in {path}", file=sys.stderr)
        return False
    if content.count(INCLUDE_OLD) != 1:
        print(f"  ERROR: '#include <cstring>' not found exactly once in {path}", file=sys.stderr)
        return False
    path.write_text(content.replace(INCLUDE_OLD, INCLUDE_NEW, 1).replace(OLD, NEW, 1))
    print(f"  input-layer-on-GPU patch applied to {path}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    paths = find_sources(Path(sys.argv[1]))
    if not paths:
        print("ERROR: src/llama-model.cpp not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(p) for p in paths) else 1


if __name__ == "__main__":
    sys.exit(main())
