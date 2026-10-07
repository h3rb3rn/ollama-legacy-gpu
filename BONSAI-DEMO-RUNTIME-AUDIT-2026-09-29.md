# Bonsai demo: inference without Ollama / llama.cpp

Source code review on 2026-09-29. Checked out locally, no setup scripts started
and no additional model weights downloaded:

- `PrismML-Eng/Bonsai-demo`: `69c3a8beeab80283bfd45cb7b7a6b927075c29fd`,
  under `../.bonsai-work/Bonsai-demo`.
- `Blaizzy/mlx-vlm`, tag `v0.7.2`:
  `a74c7de90a344a2c2c7334acb4e48b57a40480e2`,
  under `../.bonsai-work/mlx-vlm-0.7.2`.
- The Prism backend source state already present in the Ollama fork is
  `adfffbe41b2cabcd51fff326ab045662265062bb`, release
  `prism-b10743-adfffbe`. According to
  `scripts/download_binaries.sh:16`, the current demo downloads exactly this release as well.

## Actual start paths

| Entry point | Process / library | Model format |
|---|---|---|
| `start_llama_server.sh` | Prism `llama-server` | GGUF, PQ2_0 by default |
| `run_llama.sh` | Prism llama.cpp CLI | GGUF |
| `start_mlx_server.sh` with Bonsai 2 | `.venv-vlm/bin/python -m mlx_vlm.server` | separate MLX 2-bit pack |
| `run_mlx.sh` with Bonsai 2 | `mlx_generate_bonsai2.py` → `mlx_vlm.load` / `generate` | separate MLX 2-bit pack |
| `start_openwebui.sh` | Open WebUI connects to one of the two servers | depending on the chosen backend |
| `start_agent_server.sh` | wrapper around `start_llama_server.sh` | GGUF |

`start_openwebui.sh:16–25` accepts exactly `llama` or `mlx`. For `mlx`
it explicitly requires macOS/arm64. The individual MLX starters also check the
platform. A dedicated NVIDIA PyTorch/Triton/TensorRT inference path for this
27B text model is not implemented in the demo examined. This statement
concerns the demo; it is not a statement about all current or future
MLX backends or external Bonsai projects.

## What MLX executes differently

For Bonsai 2, `scripts/requirements-mlx-vlm.txt` pins MLX 0.32.2,
MLX-VLM 0.7.2 and Transformers 5.14.1. The separate `.venv-vlm` uses
the native Bonsai 2 loader. The Prism MLX fork that additionally appears in the setup
in `.venv` concerns the older 1-bit path.

`mlx_generate_bonsai2.py:40–61` requires `model_type=prism_hadamard_qwen35`,
sets the default device to `mx.gpu` and calls `mlx_vlm.load()`.
It does not load the existing PQ2_0 GGUF file but its own pack
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`.

The native implementation in
`mlx_vlm/models/prism_hadamard_qwen35/prism_hadamard_qwen35.py`:

1. Inherits the Qwen3.5 model and replaces the linear and embedding modules
   listed in the pack with Hadamard-capable quantized modules.
2. For rotated linear modules, applies sign vectors and a normalized,
   block-wise Walsh-Hadamard transformation to the input. The rotation
   computes in FP32 and converts back to the original activation data type.
3. Calls `mx.quantized_matmul` with 2 bits and group size 128. Weights
   stay packed; this is not a complete FP16 decompression of the model.
4. For embeddings, decompresses only the selected rows and, where applicable,
   transforms them back inversely.

A genuine inference path independent of llama.cpp therefore exists. It uses
MLX operations and is offered by the demo for Apple Silicon; it is not a
finished alternative CUDA server for N04 and not an interchangeable GGUF loader.

## Concrete difference from our own CUDA build: Flash Attention

The demo starts `llama-server` in `start_llama_server.sh:195` with `-fa on`.
Its CUDA build script leaves the Flash Attention compilation at the Prism default
`GGML_CUDA_FA=ON` (`ggml/CMakeLists.txt:210`).

At the time of the original review, our Dockerfiles for CUDA 12 and 13
set `ARG GGML_CUDA_FA=OFF` by default. The CUDA 11 Dockerfile also set
`-DGGML_CUDA_FA=OFF`. The RTX build was subsequently switched to
`GGML_CUDA_FA=ON`; the Bonsai CI matrix additionally passes `ON` explicitly
for CUDA 13. The legacy defaults stay unchanged.

In the actually pinned Prism source state the following chain results:

- `ggml/src/ggml-cuda/CMakeLists.txt:162`: OFF defines `GGML_CUDA_NO_FA`.
- `common.cuh:294`: as a result `FLASH_ATTN_AVAILABLE` is not defined.
- `fattn.cu:359`: the CUDA selection returns `BEST_FATTN_KERNEL_NONE`.
- `fattn.cu:587`: CUDA reports the Flash Attention operation as unsupported.
- `src/llama-context.cpp:3880`: a quantized V cache simultaneously forces
  Flash Attention unless it was explicitly disabled.

A runtime flag or `OLLAMA_FLASH_ATTENTION=true` cannot add missing compiled
CUDA kernels. The GGML scheduler checks support per
operation; CPU execution of individual operations is therefore possible despite
fully offloaded model layers.

## Runtime proof and implementation

In the previous N04 image `ollama-bonsai:rtx-validation-20260928`, the
actual GGML CUDA backend on all four original RTX GPUs reports both
F16 and Q4_0 Flash Attention as unsupported (eight negative checks).
The new `scripts/check-cuda-flash-attn.py` checks this capability directly via
the GGML C API with tensor metadata, without loading model weights.

A separate diagnostic runner with the same old image, PQ2_0 model, Q4 KV,
`-fa on` and `GGML_SCHED_DEBUG=2` actually assigns the `FLASH_ATTN` operations
to the CPU. The trace contains 336 such assignment probes from graph construction,
warm-up and execution; these are not 336 model layers. This finding proves
the CPU fallback despite `65/65` offloaded model layers. It does not yet measure
the time share of these operations.

The controlled baseline measurement with identical 128-token decode is
12.84–13.12 tokens/s (4096 context) and 13.38–13.49 tokens/s (262144 context).
The earlier roughly 1.8 tokens/s were not reproduced and must not
be used as the baseline for the speedup factor.
The measurements reserve the maximum cache but contain short prompts.

Build and the subsequent RTX validation are documented in
[BONSAI-RTX-FA-VALIDATION-2026-09-30.md](BONSAI-RTX-FA-VALIDATION-2026-09-30.md).
The production instance `11436` is not part of the test scope.

## Primary sources

- [Demo start script](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/start_llama_server.sh)
- [MLX server starter](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/start_mlx_server.sh)
- [MLX generation](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/mlx_generate_bonsai2.py)
- [Native MLX model implementation](https://github.com/Blaizzy/mlx-vlm/blob/a74c7de90a344a2c2c7334acb4e48b57a40480e2/mlx_vlm/models/prism_hadamard_qwen35/prism_hadamard_qwen35.py)
- [Prism CUDA Flash Attention selection](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/fattn.cu)
