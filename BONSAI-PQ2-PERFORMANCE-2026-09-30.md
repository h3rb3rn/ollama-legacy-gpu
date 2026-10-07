# Why PQ2_0 does not speed up in proportion to the file size

Runtime state examined: Prism `adfffbe41b2cabcd51fff326ab045662265062bb`,
Ollama v0.34.1, CUDA 13, N04 RTX test instance 11434. The production instance 11436
is not modified. The existing model tags confirm
`bonsai2:27b-pq2_0` as PQ2_0 and `qwen3.8:27b` as Q4_K_M.

## Qwen comparison on the same instance

The already available `qwen3.8:27b` was measured on the new fork:
same number-list prompt, 262144 context slots, Q4_0 K/V cache, batch 128,
temperature 0, seed 42, 128 generated tokens and the same four RTX GPUs.
The existing model-dependent chat templates were kept; Qwen has
30 prompt tokens, Bonsai 72. Only the decode phase is compared.

| Model | Run 1 tokens/s | Run 2 tokens/s |
|---|---:|---:|
| Qwen3.8:27b Q4_K_M, new comparison | 14.84 | 15.01 |
| Bonsai PQ2_0, previous RTX validation | 21.56 | 21.49 |

The mean is 14.92 versus 21.52 tokens/s, a gain of about 44 percent.
The measurements use the same configuration but are runs separated in time
and not an isolated quantization ablation. Qwen has 64 regular layers,
`n_layer_all=65`; the loader reports 66/66 GPU layers. Qwen also reserves
1152 MiB of Q4 KV per GPU and has five graph splits (Bonsai six). Its tag
additionally loads a vision projector; the test contains no image inputs.
Qwen's longer cold load time is not included in the decode rate.

Raw data: `../.bonsai-work/rtx-fa-20260929/qwen-comparison.jsonl`.
After the test Qwen was unloaded. No new weights were downloaded
and no existing model profiles were changed.

## Measured placement costs

With an identical Bonsai prompt, 4096 context slots, batch 128, seed 42 and
128 generated tokens, a single RTX 3060 in the new image reaches a mean of
28.66 tokens/s. Distribution over two RTX 2060 and two RTX 3060 reaches
21.45 tokens/s. That corresponds to about 34.9 versus 46.6 ms per token, or
25 percent less throughput. It is the combined effect of GPU selection,
transfers and synchronization; the measurement does not isolate a PCIe time share.

On the same four GPUs, 4096 and 262144 reserved context slots are
practically level with short prompts: 21.45 and 21.52 tokens/s. The pure
cache capacity therefore does not explain the current decode rate. At 256k,
however, the additional memory requirement forced the switch from the previous
single-card placement to a distribution. This was a working configuration that
is not yet optimized for the minimum number of GPUs.

## Actual PQ2 CUDA path

`ggml/src/ggml-cuda/vecdotq.cuh:978` implements
`vec_dot_pq2_0_q8_1`: the kernel reads packed 2-bit codes, unpacks them
into integer values via `__byte_perm` and computes the dot products with
`ggml_cuda_dp4a` against Q8_1 activations. Group scales are applied afterwards.
This saves weight transfers but does not automatically halve the
number of compute steps. Zero weights are not treated as skipped matrix
elements in this dense path. The weights stay packed in memory;
no complete FP16 model is materialized.

Q4 kernels also have to unpack weights. From this code finding alone
it therefore follows neither that PQ2 is slower overall nor how large its overhead
compared to Q4 is. Single-token decode uses the matrix-vector path here;
a low storage format is no proof of native 2-bit tensor-core
arithmetic.

## Additional activation transformations

`src/llama-graph.cpp:1546` inserts in `build_lora_mm` for the weights marked
in the model the matching sign/Hadamard transformation before the actual
matrix product. Transformations of the same activation are reused.
`ggml/src/ggml-cuda/fwht.cu` performs the transformation in float arithmetic.
The graph fuser in `ggml-cuda.cu:3478` can combine sign multiplication and
transformation; the transformation is already supported by CUDA.
It would therefore be wrong to name a generally missing Hadamard support
as the cause again.

Attention, recurrent state update, normalization, activations
and synchronization are not halved by the halved weight bit width either.
How much runtime these components, including Hadamard and
unpacking, take up has not yet been measured with a CUDA time profile.
The scheduler traces so far prove backend assignment, not time shares.

## Sources

- [Pinned PQ2 dot-product kernel](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/vecdotq.cuh#L978)
- [Pinned graph construction](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/src/llama-graph.cpp#L1546)
- [CUDA Hadamard kernel](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/fwht.cu)
- [Vendor model card: representation and benchmark conditions](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [Own RTX measurements](BONSAI-RTX-FA-VALIDATION-2026-09-30.md)
