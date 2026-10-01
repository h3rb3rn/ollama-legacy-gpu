# N04 production11436 Q8 KV rollout

User explicitly authorized replacement and production testing on2026-10-01.
The locally built and previously hardware-qualified image was deployed on the
original four Tesla M10 GPUs, retaining bind mount, port, restart policy,
network and Docker resource settings. No GGUF downloads/copies were made.

- Endpoint/container: `ollama-tesla-bonsai`, host port11436.
- Image ID: `sha256:c5ae3f17cfd9bc116dacc141bb5562bc92732aaf190e0ab3d3968a60f5aa24ab`.
- Build: Ollama0.34.1, CUDA12, sm50/sm52, Prismadfffbe, Flash Attention compiled ON.
- Runtime: Flash Attention true, Q8_0 cache, parallel1, native fitting,
  requested context190000, actual runner allocation190208.
- Model: existing `bonsai2:27b-pq2_0` in `/opt/ollama/models:/root/.ollama`.
- Runner confirms65/65 GPU layers, Q8 K/V, totalKV6315.50MiB,
  1578.88MiB per GPU. Compared with prior F16KV11888MiB:46.875% reduction.
- GPU memory including weights/cache/compute measured4527–4748MiB per GPU.

Production rollout and test COMPLETED. Both `deployment-result.json` and
`production-validation.json` report passed. Correct arithmetic437 and763 passed;
128-token repeated numbers benchmark1.836804 /1.837662tokens/s passed.
2048-token stability test completed at1.802195tokens/s, with no CUDA errors
or container restarts. Bonsai remains resident with190000 requested context
and24h keep-alive. Production is healthy; all other protected container IDs,
start times and restart counts are unchanged.
Cache capacity is allocated; no fully occupied190k prompt is claimed.

An initial rollout attempt was deliberately stopped after discovering that
the API reports requested190000 while the runner allocates190208. Automatic
rollback restored original containerIDa998f51cf8339c674e002d1c8ac45c2fa6958f9ca37af48618525601d56a84f0.
The corrected verifier accepts the API's requested value and independently
checks allocation in logs. The retry completed successfully; parentPID1874167
is terminal. The original container is retained stopped as
`ollama-tesla-bonsai-pre-q8-20261001-182151`, with restart disabled.

New containerID20ed4edc743a48e50b890b3e99d5312ba114a8ef59860d24f724c994585f6191.
Artifacts and rollback name: N04 `/tmp/bonsai-m10-prod-q8-20261001/`,
including `deployment-result.json`, `validation.log`, runner log and old inspect.
Dedicated reproduction config: `compose/docker-compose.n04-bonsai-m10.yml`;
also saved on N04 as `/opt/deployment/ollama/llm-studio/worker-rtx/docker-compose.bonsai-m10-prod.yml`.
Both local and host `docker compose config --quiet` checks passed. No Compose
recreation was run; the existing Docker HostConfig was preserved by the adapter.
Local raw results: `.bonsai-work/rollout-20260930/m10-production-deployment.json`
and `m10-production-validation.json`.
Stock11434/11435 and M60guard11442 are protected by identity checks.

Separate CUDA12sm50/52/60/61/70 Bonsai v0.35.0 compilation on N02 has now
finished successfully, exit0, imageIDb32e6885fd96fc827611d315424cfd44eaa73d259d141df4517beaf0e54b2c63.
It has not been hardware-qualified or deployed; this production rollout uses
the proven0.34.1 image. A subsequent M60 Q4 Bonsai test passed; M60 Q8 and
ordinary-model qualification plus GitHub activation remain outstanding.
