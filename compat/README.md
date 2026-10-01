# Ordinary Spark2.5 compatibility on Maxwell

The pinned Prism backend supports Bonsai's private PQ2/PTQ tensor formats but
does not recognize `spark2_5`. Replacing the previous upstream backend therefore
breaks an existing Spark2.5 production workload on N02. This optional CUDA12
overlay keeps that workload available while retaining the new0.35.0 Ollama API
and the complete Prism backend for Bonsai.

The Spark backend is the **previously working0.34.1 Maxwell native artifact**,
not a newly compiled upstream0.35.0 native backend. It is pinned by registry
digest in [Dockerfile.cuda12-spark](Dockerfile.cuda12-spark). Its compiled GPU
coverage is sm50/sm52 only; the dispatcher rejects other GPU architectures.
This is a narrow compatibility bridge, not general support for every model
added after the pinned Prism revision.

| Model | Native backend |
| --- | --- |
| Ordinary `spark2_5`, standard GGUF file/tensor types, Maxwell GPUs | Isolated pinned upstream-compatible backend |
| Bonsai PQ2_0/PTQ1_0 and all other models | Pinned Prism backend |
| Discovery/capability queries without a model | Pinned Prism backend |

[llama-server-dispatch.py](llama-server-dispatch.py) reads bounded GGUF metadata
and checks tensor types before allowing Spark dispatch. Private Prism file
types or tensors always stay on the Prism path. The compatibility process uses
its own library directory and excludes the primary GGML/llama library paths;
the two implementations are never linked into one process. The compatibility
directory is `/opt/ollama-spark-compat`, outside Ollama's primary backend
discovery tree. Nesting it under `/usr/lib/ollama` causes unwanted library
discovery and was rejected by the process-mapping validation.

N02 validation checked the existing Spark model at98304 context (37/37 GPU
layers), then at its active262144 context (17/37 GPU layers with F16 cache),
instruction copying and256-token generation, then switches to an ordinary
Qwen model on Prism and restores Spark. `/proc` mappings verify library isolation
per process. The original Spark model gave inconsistent arithmetic answers
under both old and new API versions; the copying probe is not an arithmetic
quality claim. Its production cache was subsequently switched to Q4 with Flash Attention
under the fleet-wide default change; see the Q4 rollout evidence. Bonsai Q4/Q8 qualification is
recorded separately in the fleet report.

The CUDA12 Actions job builds this overlay from the exact newly built Bonsai
digest and records the overlay digest for subsequent gates. CUDA11/13 do not
include the Maxwell-only compatibility artifact. A local build requires an
explicit already-qualified Bonsai base image:

```bash
docker build -f compat/Dockerfile.cuda12-spark \
  --build-arg BONSAI_IMAGE=ollama-bonsai:v035-cuda12-validation-20261001 \
  -t ollama-bonsai:v035-cuda12-spark-compat-20261001 .
python3 -m unittest discover -s compat -p 'test_*.py' -v
```
