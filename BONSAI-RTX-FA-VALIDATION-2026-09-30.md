# Bonsai RTX: CUDA Flash Attention

## Implementation and environment

`Dockerfile.cuda13-rtx` compiles with `GGML_CUDA_FA=ON`, including
quantized KV cache kernels. The Bonsai matrix in GitHub Actions passes
this value explicitly for CUDA 13. CUDA 11/12 defaults stay unchanged.
`scripts/check-cuda-flash-attn.py` checks the capability of the GGML backend
actually loaded, via its C API.

The CUDA 13 build was built successfully on N02-M60; inference and
deployment took place on N04-RTX, port 11434. Ollama v0.34.1 and Prism
`adfffbe41b2cabcd51fff326ab045662265062bb` are pinned.
Image: `ollama-bonsai:rtx-fa-cuda13-20260929`, config digest
`sha256:0f70cfaf1aba9580de10dae33f20b8580895ac28c17e94232177f9a1f182a9c4`.

Unchanged device assignment: two RTX 2060 and two RTX 3060 with 12 GB each,
UUIDs `GPU-ed954d67-584b-f101-ae1e-5cd4df31cebc`,
`GPU-6cf8a010-9ab8-f640-b5d4-b88540b220b7`,
`GPU-6e62055a-46d4-e5e1-41d6-90efc6aa3dbb`,
`GPU-63bfbd4b-9dd7-75ee-e025-d7c8966fcc42`.
Shared mount: `/opt/ollama/models:/root/.ollama`.
The production instance `ollama-tesla-bonsai:11436` was not modified.

## Proof of the operations

| Check | Old CUDA 12 image without FA | New CUDA 13 image with FA |
|---|---|---|
| Backend capability F16 and Q4_0 on four RTX | 0/8 supported | 8/8 supported |
| `FLASH_ATTN` in the scheduler trace | CPU | CUDA0 |
| Number of logged assignment probes | 336 | 336 |

The traces use the same PQ2_0 blob, the same RTX 3060, 4096 context,
Q4 KV, batch 128 and `-fa on`. The 336 probes come from graph construction,
warm-up and execution, not from 336 model layers. The message `65/65`
GPU layers alone would not have revealed this difference.

## Throughput comparison

Identical prompt: `List the integers from 1 to 200, separated by spaces.
Do not explain.` Temperature 0, seed 42, 128 generated tokens, batch 128,
one request at a time. Decode rate = Ollama `eval_count / eval_duration`.
Load time and prompt processing are not part of this rate.

| Context capacity | Old, tokens/s | New, tokens/s | Placement |
|---|---:|---:|---|
| 4096, run 1 | 13.12 | 28.45 | one RTX 3060 each |
| 4096, run 2 | 12.84 | 28.88 | one RTX 3060 each |

After the switch to native memory planning (currently deployed configuration):

| Context capacity | Old, tokens/s (one GPU) | New, tokens/s (four GPUs) |
|---|---:|---:|
| 4096, run 1 | 13.12 | 21.15 |
| 4096, run 2 | 12.84 | 21.76 |
| 262144, run 1 | 13.49 | 21.56 |
| 262144, run 2 | 13.38 | 21.49 |

At maximum cache capacity the gain is thus about 60 percent.
Distribution over four GPUs reduces the 4k decode compared to the new
single-card run; the comparison does not support adding up the GPU bandwidths.
The four outputs of the currently deployed configuration are also byte-identical
to the respective baseline runs.

The two 4k outputs are byte-identical to the respective baseline run. The new
4k run has two graph splits and 10.08 MiB of CUDA host compute buffer. The change
covers CUDA version and FA build together; the speed factor is
a comparison of the images, not an isolated timing of individual FA kernels.
The baseline runs were recorded the day before. Earlier values around 1.8 tokens/s
were not reproduced in the controlled baseline measurement.

## Finding at 262144 context slots

The first attempt with the previous placement heuristic failed with CUDA
OOM. It forces `OLLAMA_LAYER_OVERHEAD_SCALE=1.1` and places all weights
on one RTX 3060. In fact 6539.67 MiB of weights, 4608 MiB of Q4 KV,
149.62 MiB of recurrent state and up to 1313.58 MiB of compute buffer are needed.
This combination does not fit on the chosen card. The additional
CUDA buffer had not accrued there in the previous CPU fallback.

The subsequent validation therefore uses native memory planning:
GPU autodetection and forced greedy placement are disabled for this dedicated
RTX instance; the four GPU UUIDs stay explicitly assigned.
The repeated 256k requests ran through successfully. Per card, 1152 MiB of Q4 KV
and 1313.58–1323.58 MiB of compute buffer were reserved; all 65 layers
are on the GPU. The graph has six splits. The reproducible configuration
is in [docker-compose.n04-bonsai-rtx.yml](compose/docker-compose.n04-bonsai-rtx.yml).

These throughput runs check full cache capacity with short prompts, not
262144 already occupied context slots. Supplementary stability runs with
the same capacity are also complete:

| Request | Prompt tokens | Generated tokens | Decode tokens/s | Result |
|---|---:|---:|---:|---|
| 19 × 23 | 66 | 46 | 23.42 | correct: 437 |
| Color padding and question about the capital of France | 8268 | 55 | 20.75 | correct: Paris |
| Longer sorting-algorithm text | 77 | 2048 | 20.80 | token limit reached as expected |

The 8268-token prompt was processed at 310.69 tokens/s. The long
output took 101.36 seconds in total; it tests sustained decode,
not the technical correctness of a complete tutorial. None of these
requests led to OOM, runner abort or container restart. This is a
limited stability test, not a long-duration test with a fully occupied 256k KV.

Finally, `ollama:11434` is healthy and has zero restarts. ID, start time
(`2026-09-28T20:19:28.11938137Z`) and restart counter (0) of the protected
production instance `11436` are unchanged. The model was unloaded after the tests
and can be loaded again on the next request. The old
RTX container stays stopped as `ollama-rtx-pre-fa-20260929` for rollback.

The check of the model paths under `/tmp`, `/opt/ollama` and `/opt/deployment`
on N04 found only the existing 7206168928-byte Bonsai file in the shared
blob store. No weights were downloaded or copied for this validation.
On N02 no Bonsai GGUF files were found in the checked paths.

## Artifacts

Raw data and diagnostic scripts are under
`../.bonsai-work/rtx-fa-20260929/`: `before.jsonl`, `after.jsonl`,
`before-trace-summary.txt`, `after-trace-summary.txt`, `benchmark.py`,
`trace_backend.py`, `deploy.py`, `after-native-fit.jsonl`, `stability.jsonl`
and `native-fit-runtime-evidence.txt`. The complete scheduler traces are
on N04 in `/tmp/bonsai-rtx-fa-20260929/`.
The CI configuration is prepared locally; it was not pushed or
validated as a GitHub Actions run. The RTX build was actually built locally.

The cold loads of the native configuration take about 43 seconds.
The log still contains warnings from the GPU discovery watchdog, followed by
successful device detection, model load and inference. This remaining
startup delay is not fixed by the FA change.
