#!/usr/bin/env python3
"""Query the actual GGML CUDA backend for F16/Q8_0/Q4_0 Flash Attention support.

Run inside an Ollama image with GPU access. Builds tensor metadata only, without
loading model weights or executing attention. Exit 1 if any visible GPU lacks
support. Intended for the pinned Bonsai/Ollama GGML C API, not a stable ABI.
"""

import argparse
import ctypes as c
import json
import math
from pathlib import Path


class InitParams(c.Structure):
    _fields_ = [("mem_size", c.c_size_t), ("mem_buffer", c.c_void_p),
                ("no_alloc", c.c_bool)]


def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = list(args)
    return fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        help="Path to libggml-cuda.so (required if ambiguous)")
    args = parser.parse_args()
    if args.library is None:
        libraries = list(Path("/usr/lib/ollama").glob("cuda_v*/libggml-cuda.so"))
        if len(libraries) != 1:
            parser.error("Specify --library when zero or multiple CUDA backends exist")
        args.library = libraries[0]

    lib = c.CDLL(str(args.library), mode=c.RTLD_GLOBAL)
    ptr = c.c_void_p
    init = bind(lib, "ggml_init", ptr, InitParams)
    free = bind(lib, "ggml_free", None, ptr)
    tensor = bind(lib, "ggml_new_tensor_4d", ptr, ptr, c.c_int,
                  c.c_int64, c.c_int64, c.c_int64, c.c_int64)
    attention = bind(lib, "ggml_flash_attn_ext", ptr, ptr, ptr, ptr, ptr,
                     ptr, c.c_float, c.c_float, c.c_float)
    type_name = bind(lib, "ggml_type_name", c.c_char_p, c.c_int)
    count = bind(lib, "ggml_backend_cuda_get_device_count", c.c_int)
    backend_init = bind(lib, "ggml_backend_cuda_init", ptr, c.c_int)
    backend_free = bind(lib, "ggml_backend_free", None, ptr)
    supports = bind(lib, "ggml_backend_supports_op", c.c_bool, ptr, ptr)
    description = bind(lib, "ggml_backend_cuda_get_device_description", None,
                       c.c_int, ptr, c.c_size_t)
    types = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}
    for name, value in types.items():
        if type_name(value).decode() != name:
            raise RuntimeError("Unexpected GGML type enum; check the pinned C API")

    results = []
    for device in range(count()):
        name = c.create_string_buffer(256)
        description(device, name, len(name))
        backend = backend_init(device)
        if not backend:
            raise RuntimeError(f"Cannot initialize CUDA device {device}")
        try:
            for kv_type in ("f16", "q8_0", "q4_0"):
                ctx = init(InitParams(1024 * 1024, None, True))
                if not ctx:
                    raise RuntimeError("Cannot allocate GGML metadata context")
                try:
                    # Q/K/V layout: head dimension, tokens, heads, sequences.
                    # Head dimension 256 exercises the Bonsai-2 attention width.
                    q = tensor(ctx, types["f32"], 256, 1, 4, 1)
                    k = tensor(ctx, types[kv_type], 256, 256, 1, 1)
                    v = tensor(ctx, types[kv_type], 256, 256, 1, 1)
                    op = attention(ctx, q, k, v, None, 1 / math.sqrt(256), 0, 0)
                    results.append({"device": device, "name": name.value.decode(),
                                    "kv_type": kv_type,
                                    "cuda_flash_attention": bool(supports(backend, op))})
                finally:
                    free(ctx)
        finally:
            backend_free(backend)
    passed = bool(results) and all(r["cuda_flash_attention"] for r in results)
    print(json.dumps({"library": str(args.library), "passed": passed,
                      "checks": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
