# GPU release automation

Workflows build candidate images for published upstream releases. Without a
configured `OLLAMA_GPU_TEST_TARGETS` inventory, only builds run; hardware
validation, deployment, promotion and version recording stay disabled.
Invalid nonempty inventories fail before building. Complete release automation
has not yet been validated end to end on GitHub. Do not treat a local unit-test
pass or candidate build as hardware qualification or a deployed release.

## Release sequence

1. The weekly upstream check detects a published Ollama release. Failed
   attempts remain retryable because detection does not update the version file.
2. Build candidates from that Ollama version for CUDA 11, 12 and 13, with
   `BONSAI=ON`. Applying the GGUF/native compatibility patches, GGUF tests and
   compilation must all succeed. An incompatible upstream release stops here.
3. Run Bonsai plus a separate ordinary GGUF model on real K80, M10, M60 and
   RTX hardware. Maxwell and RTX must pass Q8_0 and Q4_0 tests; K80 uses F16.
   Evidence names the immutable image digest and actual hardware UUIDs.
4. Replace explicitly configured deployment targets using that tested digest.
   Preserve the model mount, GPU UUIDs, port and Docker resource settings.
   Check inference after replacement and restore the old container on failure.
5. Promote release/latest aliases after all configured deployments succeed.
   Only then update `.last-built-version`.

The full Prism backend remains pinned in `patches/bonsai/source.json`; it is
not mixed with a new upstream GGML library. The `ollama_version` in that file
records the original integration baseline, not a workflow version override.
Future API changes may require adapting patches before the failed release can
proceed. CUDA 13 cannot target Maxwell/Pascal: GTX 10 cards select CUDA 12,
whereas GTX 16/RTX select CUDA 13. The architecture table is in
`presets/gpu-targets.json`; compilation coverage and hardware test coverage
are distinct.

## Host configuration

Provision real self-hosted GitHub runners with Docker, NVIDIA Container
Toolkit, Python 3, and access to their existing Ollama model directory.
Use a host-specific runner label; no label or K80 host is assumed by this repo.
The runner must be reserved for trusted repository workflows.

Repository variable `OLLAMA_GPU_TEST_TARGETS` contains a JSON array. Each row
has `id`, `family` (`k80`, `m10`, `m60`, `rtx`), `variant`, `runner` (an array
of actual runner labels including `self-hosted`), `config` (an absolute path
to a host-owned JSON file), and an explicit boolean `deploy`. All four
hardware families must be present. `deploy=false` means hardware validation
only; it does not authorize replacement of that host's services.

Each host JSON file contains:

```json
{
  "id": "n04-m10",
  "family": "m10",
  "variant": "cuda12-maxwell",
  "gpu_uuids": ["REPLACE_WITH_THE_VERIFIED_GPU_UUIDS"],
  "model_root": "/opt/ollama/models",
  "context": 190000,
  "bonsai_model": "bonsai2:27b-pq2_0",
  "ordinary_model": "qwen3.5:4b",
  "protected_containers": ["ollama", "ollama-tesla-bonsai", "ollama-rgtx", "ollama-m60-guard"],
  "deployment": {
    "container": "ollama-tesla-bonsai",
    "port": 11436,
    "bind_ip": "",
    "kv_type": "q8_0"
  }
}
```

This is a schema example, not an installed target. Supply the complete actual
GPU UUID list. M10/M60 tests request 190000 context; RTX requests 262144.
The optional `deployment` object is required only when `deploy=true`.
The test creates a temporary localhost-only container, reuses existing blobs,
and leaves production containers running. GPU pools must be idle during the
test window; busy pools fail instead of unloading someone else's model.

The two arithmetic checks, two 128-token speed measurements and 2048-token
stability output run for each cache type. Checks require the requested KV
format in runner logs and full GPU layer placement, and reject CUDA errors.
The large context is allocated; these tests do not fill all 190k/256k slots.

Production replacement requires the successful test artifact and an unchanged
host-config hash. The current adapter supports one shared writable bind mount,
the standard `serve` command and a bridge-network port mapping. Custom setups
are rejected before any stop. The previous container is retained with restart
disabled after success; on failure its name and restart policy are restored.
Unexpected host loss or SIGKILL during replacement still requires operator
recovery from the retained container. Model files are never removed.

## Current remaining activation work

- M10 Q8_0 and Q4_0 at requested context 190000 passed both Bonsai and
  ordinary-model tests, including 2048-token decode. CUDA-11 v0.34.1 compilation
  passed; K80 hardware inference remains outstanding. Actual Bonsai M60 Q4 at
  requested190000 context now passed arithmetic, repeated decode and2048-token
  stability on the original two GPUs; the original production service was
  restored afterward. M60 Q8 inference and ordinary-model qualification remain.
  The broader v0.35.0 CUDA-12 build finished successfully but lacks hardware tests.
- Provide/verify the K80 host and actual GitHub runner labels.
- Restore GitHub CLI authentication on the control host.
- Confirm the auto-deployment target inventory, then install the host configs.
- Real Docker replacement and rollback passed using disposable containers on
  N04. GPU requests, mount, port, resource limits and restart settings were
  preserved. The rollback test deliberately injected inference failure; it
  did not load an LLM or constitute a production release qualification.
- User directs Stock Ollama 0.35.0 for N04 ports 11434 and 11435. Exclude these
  endpoints from automatic fork deployment unless that instruction changes.
- Publish the reviewed changes and observe a complete Actions run, including
  hardware artifacts, rollout, promoted digests and version recording.

Workflow validation uses `actionlint`; Python regression checks run with
`python3 -m unittest discover -s tests -v`. GitHub runner labels and job
dependencies follow the [GitHub runner documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)
and [job-output documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/pass-job-outputs).
