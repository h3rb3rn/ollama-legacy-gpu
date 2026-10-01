# Ollama 0.35.0 fleet rollout — 2026-10-01

All14 endpoints report0.35.0: twelve Tesla fork services and two official
Stock services. All twelve fork containers are Docker-healthy, with zero
restarts; the Stock APIs are reachable and their container identities unchanged.
GPU reservations, ports, resource limits, log settings, network modes,
default-context settings and shared model
mounts were checked against the original snapshots.

| Host | Service:port | Runtime | KV cache | Default context |
| --- | --- | --- | --- | --- |
| N04-RTX | ollama:11434 | Stock | q4_0 | 262144 |
| N04-RTX | ollama-rgtx:11435 | Stock | q4_0 | 262144 |
| N04-RTX | ollama-tesla-bonsai:11436 | Fork | q4_0 | 190000 |
| N04-RTX | ollama-m60-guard:11442 | Fork | q4_0 | 32768 |
| N11-M10 | ollama:11434 | Fork | q4_0 | 131072 |
| N02-M60 | ollama-m60-pool:11434 | Fork | q4_0 | 131072 |
| N02-M60 | ollama-m60-gpu4:11435 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu5:11436 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu6:11437 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu7:11438 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu8:11439 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu9:11440 | Fork | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu10:11441 | Fork + isolated Spark backend | q4_0 | 65536 |
| N02-M60 | ollama-m60-gpu11:11442 | Fork | q4_0 | 65536 |

Contexts above are server defaults; API requests can override them. Existing
loaded models were restored. At the user's subsequent request, every Tesla
service now uses Q4 KV and enabled Flash Attention. The initial F16/Q8 rollout
records are retained as historical evidence; final Q4 records are separate.
N11 Qwen35b at190k now uses42/42 GPU layers (previously41/42 with F16).
This does not prove every graph operation executes on GPU.

## Native Bonsai qualification

Ollama sourcev0.35.0 commitcc4069396f3ad2c370c53eed2e4a42ac13adab84;
Prism commitadfffbe41b2cabcd51fff326ab045662265062bb; CUDA12 FAON,
compiled CC5.0/5.2/6.0/6.1/7.0. Published Bonsai27b PQ2_0 weights are unchanged.
Tests allocate cache capacity, not a completely occupied190k/256k prompt.

| N04 GPUs | KV | Allocated context | KV GiB | 128-token decode tok/s | 2048-token stability tok/s |
| --- | --- | --- | --- | --- | --- |
| 4 M10 | Q8 | 190208 | 6.17 | 1.213 /1.806 | 1.648 |
| 2 M60 | Q4 | 190208 | 3.27 | 7.319 /7.357 | 7.079 |
| 2 M60 | Q8 | 131072 | 4.25 | 7.333 /7.313 | 7.102 |

All cases passed actual CUDA F16/Q8/Q4 capability probes, arithmetic437/763,
65/65 model-layer GPU offload and2048-token generation. M60 Q8/190k was stopped
because only53/65 layers fit on those two GPUs; it is not qualified.
M60 ordinary Qwen4b Q4/Q8 stability passed at18.191/18.375tok/s.
M10 Q4 on0.35 passed the final256-token production rollout at1.798tok/s
with65/65 GPU layers and3.27GiB KV. Its2048-token Q4 qualification remains
the historical0.34.1 result; the new0.35 long test above is Q8.
RTX Q4/256k results are also historical0.34.1; final RTX services run Stock.

The first M10 qualification completed successfully, but the simultaneous
authorized M60 guard replacement invalidated the protected-container snapshot.
The final M10 replacement was serialized, reused the exact native image/GPU/
cache proof, and repeated capability, correctness, KV/log and256-token checks
(1.793tok/s) with all protected identities stable. This was a transaction
serialization correction, not a failed native Bonsai stability test.

Every replacement also passed an ordinary loaded-model256-token regression
or the Bonsai smoke test. N02 GPU4's arithmetic answer447 matched its old
baseline; it is not claimed correct. The Spark probe uses instruction copying
because arithmetic varied on the old backend too.

## Spark compatibility bridge

The pinned Prism backend lacks spark2_5. N02:11441 therefore runs the0.35.0 API
plus an isolated, previously working0.34.1 Maxwell native backend for ordinary
Spark only. Bonsai and other models use Prism. This is explicitly not a newly
compiled upstream0.35.0 Spark backend.

The fallback's complete libraries live outside discovery at
/opt/ollama-spark-compat. Private Prism file/tensor types cannot route there,
and GPU support is limited to CC5.0/5.2. Real /proc mappings passed for both
Prism Qwen and Spark processes. An earlier nested-library overlay was rejected,
rolled back and replaced by the isolated image.
See [implementation and tests](compat/README.md).

## Artifact identity and persistence

The transferred native image archive SHA256 was
a70dfad4250c5f2b2bc1ee7b0aebc66c4112069b7f2ddaa39706b4ff1dcbd931
on all three hosts. Loaded native image IDs:
N02/N11 b32e6885fd96fc827611d315424cfd44eaa73d259d141df4517beaf0e54b2c63;
N04 1170395e1eca4bc1c330642c2b63118bb32f3c029b20342c64420b05516b7485.
N02 Spark overlay:
a6a36d8f9625755ff8898ab5457386d6f2503ffcc794ad4cc83c5c658dfd3366.
Stock registry digest:
2a6e883b917fc543389599dae79918f5cac9e1438890506982f44aa4f5625d01.

Canonical per-host Compose snapshots are stored at
/opt/deployment/ollama/fork/compose/docker-compose.fleet-v035.json.
All passed Compose validation. N02 worker-m60 source and env profiles, N11
worker-tesla source and N04 M10 production profile were updated, with
.pre-v035-20261001 backups. Old stopped production containers remain for rollback.
Q4 source-profile backups use .pre-q4-20261001. N04's guard source now
extends its exact canonical installed profile, preserving the bridge network.
All services use the existing /opt/ollama/models:/root/.ollama pool on each host.
No new Bonsai weight copy was deployed; temporary image archives were removed.

## Validation and Actions

Local validation:33 release/validation tests plus8 compatibility tests passed;
both Actions workflows passed actionlint; git diff whitespace checks passed.
The pushed upstream-sync run is
[36921557374](https://github.com/h3rb3rn/ollama-legacy-gpu/actions/runs/36921557374).
The final source push triggers a new run with the isolated CUDA12 overlay.

Actions builds CUDA11/12/13 candidates. CUDA11 hardware remains untested.
Without a configured GPU inventory these are candidate builds, not automatic
production promotion; .last-built-version remains the last fully gated version.
The running fleet uses the locally built and hardware-qualified immutable
artifacts above, not an unqualified moving CI tag.

[Machine-readable evidence](docs/evidence/fleet-v035-20261001/README.md) contains
individual measurements, preservation checks, backend mappings and final IDs.

## Final Q4 default rollout

At the user's explicit request all twelve Tesla fork services were replaced
with the same immutable0.35.0 images, Q4 KV and Flash Attention enabled.
The final audit confirms all14 APIs remain0.35.0, all twelve Tesla services
healthy with zero restarts, and the two Stock container identities unchanged.

Every Tesla case passed the actual CUDA capability probe, existing-model
correctness/baseline check and256-token generation. Actual K(q4_0)/V(q4_0)
allocation was required in each runner log. This does not prove every graph
operation executes on GPU and is not a full-cache endurance or broad quality test.

| Endpoint | Regression model | Requested context | Decode tokens | tok/s | Logged layer placements |
| --- | --- | --- | --- | --- | --- |
| N04-RTX:11442 | qwen3.5:4b | 32768 | 256 | 19.094 | 34/34 |
| N04-RTX:11436 | bonsai2:27b-pq2_0 | 190000 | 256 | 1.798 | 65/65 |
| N11-M10:11434 | qwen3.6:35b | 190000 | 256 | 6.913 | 42/42 |
| N02-M60:11434 | ornith:9b | 131072 | 256 | 14.032 | 33/33 |
| N02-M60:11435 | qwen3.5:4b | 262144 | 256 | 18.949 | 34/34 |
| N02-M60:11436 | granite4.2:3b | 131072 | 256 | 26.288 | 41/41 |
| N02-M60:11437 | granite4.2:3b | 20480 | 256 | 26.263 | 41/41 |
| N02-M60:11438 | gemma3:4b | 131072 | 256 | 19.928 | 35/35 |
| N02-M60:11439 | nemotron-3-nano:4b | 262144 | 256 | 26.337 | 43/43 |
| N02-M60:11440 | huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF:latest | 262144 | 256 | 19.916 | 34/34 |
| N02-M60:11441 | hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M | 262144 | 256 | 19.850 | 37/37, 34/34, 37/37 |
| N02-M60:11442 | hf.co/webAI-Official/TwIL-LM3-Pro:Q4_K_M | 131072 | 256 | 26.543 | 41/41 |

Spark's log includes the temporary ordinary Qwen backend isolation probe as
well as Spark before/after restoration. At262144 context Spark now fits37/37
GPU layers, compared with17/37 in the initial F16 rollout;256-token decode
increased from10.965 to19.850tok/s. Its copy probe and real library isolation
passed again. N11 Qwen35b now fits42/42, compared with41/42 previously.
Q4 reduces KV memory but does not imply a uniform speed gain across models.

All canonical and original production profiles were resolved through Docker
Compose to confirm Q4/FA defaults. Other environment values, image IDs, GPU
requests, resource/log limits, networks and shared mounts were checked against
pre-Q4 snapshots. Stopped pre-Q4 containers remain for rollback; restoring a
profile requires its matching pre-Q4 backup as well.

[Q4 rollout proof](docs/evidence/fleet-v035-20261001/q4-default-rollout.json),
[final Q4 audit](docs/evidence/fleet-v035-20261001/q4-final-audit.json) and
[Spark library isolation](docs/evidence/fleet-v035-20261001/q4-backend-isolation.json)
record the final container IDs and settings.
