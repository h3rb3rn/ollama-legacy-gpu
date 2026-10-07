# Bonsai 2 27B on Tesla M10/M60 in the Ollama fork

As of: 2026-09-28. Result: **technically plausible and portable, but not yet supported in the
fork examined and not validated as a complete model on Maxwell.** A backend port including an Ollama importer adaptation is required.

## Basis of the review

- Local packaging repository: `h3rb3rn/ollama-legacy-gpu`, HEAD `0bd466c`.
- Local Ollama sparse checkout: `../ollama-src`, HEAD `e434a93`,
  `LLAMA_CPP_VERSION=b9672`. Missing importer files were read from exactly this
  Git state. This is not a statement about the image running in production.
- Bonsai-demo `main`: at retrieval `9ef32054fe44797376792c869891163083d64bd0`.
- Native implementation examined: Prism release `prism-b10709-9a9394a`,
  the release base named in `BACKEND-SUPPORT.md`. No moving branch as
  the implementation basis. `prism-v7` pointed to
  `c1abda39458458ebfb4ec0722bd2224aab26e680` at retrieval, but was not tested instead of
  the release base.
- Read directly: the first 16 MiB of the published PQ2_0 GGUF; the complete
  metadata and tensor directory region ends at byte 11,120,982.
- No complete model weights downloaded, no inference on a
  Tesla host, no change to running services or production images.

## What has to be adopted

The demo mainly provides installation and start scripts. The decisive
implementation lives in the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp).
The model weights are ternary and additionally stored in a rotated basis.
The mathematically matching activation transformation is part of inference.

Direct finding from the GGUF:

- Architecture `qwen35`, 64 layers, embedding width 5120, FFN width 17408.
- Full attention every four layers; maximum declared context 262144.
- 851 tensors: 402 of type 142 (`PQ2_0`), 353 F32, 96 BF16.
- `general.file_type=141` — FileType and TensorType have different IDs.
- `prism.hadamard.version=1`, block size 1024, normalized Sylvester-Walsh-
  Hadamard transformation, explicit signs.
- 401 transformed weight matrices; additional inverse transformation after
  the lookup of `token_embd.weight`.
- Sign vectors for widths 5120, 6144 and 17408; 28672 entries in total.
- `prism.hadamard.gdn_v_grouped=true`: the feature order in the Gated DeltaNet
  path must also be taken into account.

A single FWHT kernel or enabling a type number is therefore not sufficient. Required
are loader, graph integration, signs, inverse embedding
transformation and GDN ordering. The implementation lives in particular in
`src/llama-model.{cpp,h}`, `src/llama-graph.{cpp,h}` and the GGML backends.
The integration must also support CPU offload correctly.

Prism explicitly warns: stock llama.cpp rejects PQ2_0/PTQ1_0 as unknown types.
The separate Q2_0 development file, by contrast, can load and without the
transformations produces unusable text. Successful loading is no proof of
correctness. See the [backend matrix](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).

## Which "Q2" is meant

The sizes come from the Hugging Face file API; GB is decimal, GiB binary.

| Variant | Weight packing | File size | Classification |
|---|---|---:|---|
| PQ2_0 | 128 values in 34 bytes, 2.125 bit/weight | 7,206,168,928 bytes / 6.71 GiB | The matching 2-bit target in the linked main repository |
| PTQ1_0 | 128 values in 28 bytes, 1.75 bit/weight | 5,946,648,928 bytes / 5.54 GiB | The same ternary weights packed more densely; more memory reserve |
| Q2_0 | 64 values in 18 bytes, 2.25 bit/weight | 7,626,008,928 bytes / 7.10 GiB | Separate development file; Prism transformations still required |

PQ2_0 is neither Q2_K nor IQ2. Renaming the type number or a normal
`ollama create --quantize Q2_K` does not establish Bonsai support.
Source: [formats](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md),
[model files](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main),
[Q2_0 development file](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf-dev/tree/main).

## Maxwell: supportable operations and bottlenecks

Build targets of the existing fork include `50-real;52-real`.
The Dockerfile uses CUDA 12.0.1. This base is sensible for a port;
CUDA 13 removes offline compilation for Maxwell. The blanket statement
in the existing README that CUDA 12 excludes Maxwell is not correct.
See [NVIDIA CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

The Prism code examined contains the following paths:

1. `ggml/src/ggml-cuda/fwht.cu`: F32 arithmetic, warp shuffles and shared memory,
   including signs and width 1024. No mandatory tensor cores.
2. `mmvq.cu`: quantized matrix-vector path also for PQ2_0/PTQ1_0; regular
   Maxwell path for up to eight output columns. `common.cuh::ggml_cuda_dp4a`
   contains a scalar fallback for devices without hardware DP4A.
3. `mmq.cu::ggml_cuda_should_use_mmq`: the quantized matrix-matrix path is rejected
   below the DP4A architecture; PTQ1_0 MMQ needs the Turing MMA path on NVIDIA.
   Merely setting FORCE_MMQ does not remove this limit,
   because the architecture check comes first.
4. For larger prompt batches, `ggml-cuda.cu::ggml_cuda_mul_mat_cublas`
   therefore typically follows. On Maxwell the affected quantized matrix is
   temporarily dequantized to F32 and computed with cuBLAS.
   The weights fundamentally stay packed; an additional
   working buffer arises per affected operation.

There is thus **no fundamental Maxwell prohibition recognizable from the core
arithmetic**. But there is also no proof of the advertised modern
GPU speed. Prompt processing and decode must be measured separately. No reliable
tok/s forecast without a hardware test.

The default architectures of `scripts/build_cuda_linux.sh` start at SM80;
explicit build targets are needed for Maxwell. The script also clones `prism`,
while the format documentation assigns this branch to the older format generation.
For a reproducible port, therefore explicitly obtain the reviewed
release commit and do not use the default clone unchecked.

## Memory and a sensible starting configuration

A PQ2_0 file size of 6.71 GiB is no guarantee of operation on an 8 GiB GPU.
In addition there are KV cache, recurrent state, CUDA context and compute
buffers. From the GGUF geometry, about 64 KiB per token follow for the F16 KV
cache: 4096 tokens about 256 MiB, 8192 tokens about 512 MiB; additional states
and working buffers are not included.

An FFN weight matrix has 5120 × 17408 elements. Its F32 intermediate representation
in the cuBLAS path alone takes up 340 MiB. The output matrix is larger:
5120 × 248320, with complete F32 dequantization about 4.74 GiB. That is a
conditional risk with many simultaneously requested output rows, not a
blanket additional requirement at every decode step. Small output batches can
take the quantized MMVQ path. The normal embedding lookup
also does not dequantize the entire embedding table.

Recommended start of validation: text-only, one parallel request, 4096 context,
PQ2_0 initially on two free GPU dies with about 8 GiB each. Then
compare the single-GPU limit, batch sizes and PTQ1_0. The memory of several dies
of an M10/M60 board is not shared memory; layer distribution is necessary.
Two GPUs are a working hypothesis with more reserve, not an approval measured here.
Vision would require further model and compute buffers.

Validate Flash Attention separately. The existing Dockerfile leaves
`GGML_CUDA_FA=OFF` as the default and documents problems of large multi-GPU pools.
The single-GPU approval there does not automatically prove Bonsai compatibility.
The new Prism path also has to be checked with and without Flash Attention.

## Concrete changes in the Ollama fork

| Area | Required change |
|---|---|
| GGML/llama.cpp | Adopt PQ2_0 type traits, CPU and CUDA operations and the complete Hadamard model contract; optionally carry PTQ1_0 |
| Ollama `fs/ggml/type.go` | Add tensor IDs 142/143 and FileType mappings; Q2_0 only if an additional variant is deliberately wanted |
| Ollama `fs/ggml/ggml.go` | Correct block and type sizes: PQ2_0 128/34, PTQ1_0 128/28; do not treat unknown types as zero size |
| Import and API | Preserve original GGUF metadata; check end of file, size, model information and import |
| Backend build | Build a complete compatible native payload from the same source state, including CPU and CUDA; do not mix a foreign CUDA .so into the old GGML payload |
| Ollama compatibility | Port the hook patch to the chosen Prism state or backport Bonsai changes onto Ollama's pinned base |
| Existing fork patches | Reapply and test GPU selection, layer distribution, batch and Flash Attention adaptations |
| Model behavior | Check chat template, stop tokens, thinking and then tool calling |

In the Ollama state examined, PQ2_0/PTQ1_0 are not defined. For unknown
tensor kinds `TypeSize()` returns 0 and `BlockSize()` 256; the GGUF reader uses
`tensor.Size()` to determine offsets and end of file. The importer is thus
a real integration point, not just a missing label in the UI.

`LLAMA_CPP_FORK` is explicitly not wired up in the existing Maxwell Dockerfile.
Merely setting a different repository URL as a build argument is not enough. In addition, the local
Ollama hook patch was checked with `git apply --check`
against the Prism release: it fails at `src/llama-model-loader.cpp`
and `tools/mtmd/clip.cpp`. The complete backend swap is therefore not directly
applicable. Changes in Ollama's further architecture overlay must also be
taken into account; this dry run checks only the named hook patch.

## Checks performed and open evidence

- Read repository, commit states, existing local changes and build files.
- Parsed GGUF metadata and all 851 tensor directory entries directly.
- Traced type sizes, graph transformations, CUDA dispatch and F32 fallback
  in the source code.
- Ollama hook patch dry run: **failed**, conflict points identified.
- CUDA compile tests: see the following results section.
- Still open: complete CUDA 12.0.1 build, linking the Ollama payload,
  GGUF import test, CPU/GPU correctness comparison, real M10/M60 inference,
  prompt/decode benchmark, memory maximum and multi-GPU stability.

For the implementation, first validate the native Prism runner in isolation on Maxwell,
then its integration into Ollama. Compare correctness using
identical prompts and logits/perplexity within defined numerical
tolerances; readable text alone is not sufficient as a criterion.
Only then consider an approval for the production fork image.

## Result of the CUDA compile tests

All six translation-unit compilations were successful (exit 0,
empty compiler logs):

| Source file | sm_50 | sm_52 |
|---|---|---|
| `ggml/src/ggml-cuda/fwht.cu` | passed | passed |
| `ggml/src/ggml-cuda/convert.cu` | passed | passed |
| `ggml/src/ggml-cuda/mmvq.cu` | passed | passed |

The image `nvidia/cuda:11.8.0-devel-ubuntu22.04`, already present locally, was used,
without network and without GPU access.
The source tree was mounted read-only. Compiler call per combination:

```sh
nvcc -std=c++17 -arch=sm_50 -Iggml/include -Iggml/src \
  -c ggml/src/ggml-cuda/fwht.cu -o /out/fwht-sm50.o
```

Analogously for `sm_52`, `convert.cu` and `mmvq.cu`. This proves compilable
device code for the selected core operations on both target architectures.
It is explicitly not a complete backend build, not a CUDA 12.0.1 test
and not a runtime or accuracy test. The remaining translation units and
the interplay with Qwen35/GDN remain to be checked.

Temporary working files of this session: `/tmp/bonsai-prism-audit`,
`/tmp/bonsai-cuda-smoke`, `/tmp/bonsai-gguf-audit.json` and
`/tmp/bonsai-inspect-gguf.py`. SHA-256 of the release source archive checked:
`bd1fd5c4a5fd554ef8e9168472338f45fe598b94bba8810a233aca89c7d00cd1`.

## Implementation and hardware addendum (2026-09-28)

The implementation previously described as open has meanwhile been added to the Ollama fork:
pinned Prism source, Ollama loader compatibility, PQ2_0/PTQ1_0
GGUF types and sizes, import helpers and a CUDA 12.0.1 image for `sm_50` and
`sm_52`. The complete build on N02-M60 is successful. The GGUF parser
tests passed.

On N04-RTX, Ollama `0.34.1` ran with four Tesla M10 in a single
instance (`ollama-tesla-bonsai`, port 11436). The model copy was verified with SHA-256
`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`
and imported into the pool `/opt/ollama/models` already used by other containers.
`OLLAMA_SCHED_SPREAD=true` distributed all 65 model layers to CUDA0–CUDA3 at
4096 context. A chat inference correctly returned `437` to
the question about 19×23.

An inference also ran with `num_ctx=262144`; the memory planner loaded
53/65 layers onto the GPU and left 12 layers in CPU RAM. After reduction to
`num_ctx=190000`, the context was internally rounded to 190208 tokens. Ollama
reserved 2972 MiB of KV per M10 for it; the model loader reported 65/65 layers on
the GPU. The answer was again `437`. The short answer with four generated
tokens measured 1.03 prompt tokens/s and 0.45 decode tokens/s. A longer
throughput run at 190k generated 64 tokens: 0.20 prompt tokens/s with 42
prompt tokens and 1.84 decode tokens/s. The prompt value fluctuated markedly
between the short runs; both phases are therefore listed separately.

This proves loading and text inference in the fork as well as distribution over four
Maxwell GPUs, but not yet a numerical CPU/GPU logit comparison or a
completeness check of the model behavior (e.g. tool calling). Speed
stays low at large context; this is no proof of production performance.

## RTX and CI addendum (2026-09-29)

The RTX full-context run on N04 passed with Q4 KV at `num_ctx=262144`. With
a short 30-token prompt, 13.16 prompt tokens/s and 1.78 decode
tokens/s were measured; the request ran through completely. The KV cache occupied
4608 MiB, the Ollama API reported about 11.97 GB of VRAM for the model. This checks
reservation and stability at maximum context capacity, but not a
262k-long prompt.

The four RTX cards assigned to the instance became visible, but the fitter
loaded all 65 model layers onto one RTX 3060. During decode the GPU
utilization was only about 15 percent, while the N04 host ran with four vCPUs and
high base load. A separate Qwen3.5-4B control model reached on the
same host only about 1.19 decode tokens/s with about 2.5 CPU cores of
runner utilization. These observations do not determine the cause of the bottleneck.
CPU utilization can arise, among other things, from CUDA calls or busy waiting; it proves neither CPU compute work on the model nor a limitation
of the GPU by the CPU. Using only one RTX is also not by itself a
performance defect if model and cache fit on it.

Correction of the previous diagnosis: the nominal memory bandwidths of the
four M10 must not simply be added up for a single decode stream and
compared with one RTX. With layer splitting, consecutive
model sections run on different GPUs; in addition, transfers and synchronization
occur. Likewise a PCIe bottleneck is not demonstrated without
transfer and timing measurements. The runs so far had
different context sizes and are not a controlled comparison.

The Prism source state examined explicitly keeps the input layer on the CPU
(`src/llama-model.cpp`); the message `65/65` counts the offloadable layers.
Host calls for CUDA graph launch and synchronization also stay in the
runner. This refutes a completely CPU-independent execution, but
does not yet explain the low measured speed. To determine the cause,
a time-correlated CPU/CUDA profile during decode is missing,
distinguishing kernel runtime, host wait time and memory transfers.

The Actions now build Bonsai in separate images bound to Ollama v0.34.1
for CUDA 11/K80, CUDA 12/M10-M60 and CUDA 13/RTX. The existing rolling
production tags stay separate from this. This is a workflow and Dockerfile
change; the three new container images have not yet been built or
published here. `ollama:11436` is now a verified
production instance and was not modified during this investigation.

## Verified RTX correction (2026-09-30)

The subsequent scheduler trace proves CPU execution of
`FLASH_ATTN` in the old image. With the CUDA 13 image actually built and
`GGML_CUDA_FA=ON`, these operations run on CUDA. With an identical
4k single-card workload, decode rises from about 13 to 28–29 tokens/s.
The earlier 1.78 tokens/s were not reproduced in the controlled baseline run
and are not a suitable denominator for the speedup factor.

At 262144 context slots, GPU Flash Attention needs additional CUDA buffers.
The previous blanket placement heuristic underestimates this need and
causes OOM on a 12 GB card. With native memory planning on the four
originally assigned RTX GPUs, the new fork runs successfully on N04:11434
at 21.49–21.56 tokens/s compared with 13.38–13.49 in the baseline run. The cache
capacity is reserved to the maximum; this is no benchmark with 256k occupied
prompt tokens. Details: [RTX validation](BONSAI-RTX-FA-VALIDATION-2026-09-30.md).
