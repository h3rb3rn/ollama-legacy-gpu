# N04 Stock Ollama 0.35.0 deployment

Explicit user instruction replaces the final fork deployment on ports 11434
and 11435 with official Stock Ollama 0.35.0. Both deployments passed API,
configuration-preservation and real inference checks on 2026-10-01.

Official image: `ollama/ollama:0.35.0`, pinned digest
`sha256:2a6e883b917fc543389599dae79918f5cac9e1438890506982f44aa4f5625d01`.

| Endpoint | GPUs retained | Validation |
| --- | --- | --- |
| `ollama:11434` | Original two RTX 2060 and two RTX 3060 | Qwen3.8:27b Q4_K_M, 66/66 GPU layers, correct 437 |
| `ollama-rgtx:11435` | Original RTX 2060 and GTX 1060 | Existing Sovereign Planner 9b, 33/33 GPU layers, correct 437 |

Both use allocated context 262144, Q4_0 KV and enabled Flash Attention.
These checks did not fill all 256k context slots. Stock automatically selects
CUDA 12 for the mixed RTX/GTX pool, retaining both devices. Original GPU UUIDs,
port bindings, restart policies, resource limits and writable model bind
`/opt/ollama/models:/root/.ollama` were preserved. The main container remains
on the Docker bridge; the planner retains `worker-rtx_ai-stack-network` and
its existing aliases. No GGUF was downloaded or duplicated.

Stock does not include the private Prism PQ2_0/PTQ1_0 tensor extensions used
by Bonsai. Protected production `ollama-tesla-bonsai:11436` retains its fork,
container ID, StartedAt and zero restart count, and remains healthy. M60 guard
11442 was likewise unchanged.

Qwen repeated numbers-prompt benchmark on 11434, seed42, temperature0,
batch128, 128 generated tokens, context262144: **14.6097 / 14.6455 tokens/s**.
Both outputs match. First benchmark load took95.14s; warm reload4.04ms.
The initial deployment arithmetic request took5m51s including first model
initialization/vision warmup; GPU discovery logged a watchdog timeout.
The slow first arithmetic decode measurement is not used as the repeatable
speed result. No CUDA OOM, illegal-access error or container restart occurred.
Planner arithmetic emitted449 tokens at23.31 tokens/s; this is a smoke check,
not the same workload as the Qwen benchmark. Planner remains loaded at256k;
Qwen was unloaded after its tests.

Retained stopped rollback containers, restart disabled:

- `ollama-pre-stock-20261001-114931`
- `ollama-rgtx-pre-stock-20261001-114931`

Host Compose source `/opt/deployment/ollama/llm-studio/worker-rtx/docker-compose.yml`
now pins these two services to official0.35.0, removes the main service's local
image build and uses environment files `.env.stock-rtx` / `.env.stock-rgtx`
with the deployed runtime settings. `docker compose config --quiet` passed.
Source backup: `docker-compose.pre-stock-20261001.yml`. Other service definitions
were untouched; no full-stack Compose recreation was performed.

Deployment transaction and raw evidence are on N04 under
`/tmp/bonsai-stock-20261001/`: `stock-deployment-result.json`,
`qwen-benchmark.jsonl`, `benchmark.exit` (0), runner logs and original inspect
snapshots. Local copies are under `.bonsai-work/rollout-20260930/`.

For image installation only the unreferenced official0.24.0 Docker image was
removed; existing fork rollback images and the shared model pool were retained.
N04 root free space increased from9.3GiB to45GiB after this and stock installation.

The separate Bonsai v0.35.0 CUDA12 build on N02 remains in progress; GitHub
Actions activation and K80/M60 hardware qualification are still outstanding.
Neither Stock endpoint is authorized as an automatic fork rollout target.
