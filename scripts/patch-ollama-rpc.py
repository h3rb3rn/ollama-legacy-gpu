#!/usr/bin/env python3
"""
patch-ollama-rpc.py — Patches llm/llama_server.go so that one Ollama instance
can use the GPUs of other hosts through the llama.cpp RPC backend.

Opt-in via OLLAMA_RPC_SERVERS=host:port[,host:port...] on the head instance.
Each entry is an `rpc-server` (built with -DGGML_RPC=ON, shipped in this image
under /usr/lib/ollama/rpc-server) that exports the GPUs of its host. Without
the variable nothing changes.

What the patch does:
  1. Appends `--rpc <servers>` to the llama-server command line of a model
     load. The servers are deliberately NOT passed as LLAMA_ARG_RPC in the
     container environment: that variable would also reach the
     `llama-server --list-devices` call of the discovery, where the remote
     devices carry no compute capability and would be dropped (or would break
     the CUDA_VISIBLE_DEVICES filtering of the local devices).
  2. Returns early from selectGPUPool() (patch-ollama-dynamic-pool.py):
     llama.cpp puts RPC devices in front of the local ones, so the device list
     is [RPC..., CUDA0..CUDAn-1]. The fast-pool restriction, the --main-gpu
     index and the layout-cache --tensor-split all count local CUDA devices
     only and would point at the wrong device.

Layer placement is left to fit.cpp. With OLLAMA_FORCE_GPU_LAYERS the greedy
fill of patch-llama-tier-fitting.py starts at the highest device index, i.e. it
fills the local GPUs first and the remote ones after that: the hidden state
crosses the network once per decoded token.

Must run after patch-ollama-dynamic-pool.py.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [OLLAMA_RPC_SERVERS patch]"
TARGET_FILE = "llm/llama_server.go"

# Anchor 1: the model launch, same line the pool patch inserts in front of.
LAUNCH_ANCHOR = "params = appendFlashAttentionArgs(params, launch.gpus)"

# Anchor 2: inside selectGPUPool(), after the scale override and the main-GPU
# index, in front of the fast-pool decision.
POOL_ANCHOR = '\tfastDevices := os.Getenv("OLLAMA_FAST_GPU_DEVICES")\n'

LAUNCH_CODE = '''// [OLLAMA_RPC_SERVERS patch] cross-host GPU pool via the llama.cpp RPC backend
\tif _rpc := llamaServerRPCServers(); _rpc != "" {
\t\tparams = append(params, "--rpc", _rpc)
\t\tslog.Info("cross-host GPU pool: adding remote devices via llama.cpp RPC", "servers", _rpc)
\t}
\t'''

POOL_CODE = '''\t// [OLLAMA_RPC_SERVERS patch] RPC devices precede the local CUDA devices in the
\t// llama.cpp device list: indices and counts derived from CUDA_VISIBLE_DEVICES
\t// (--main-gpu, fast pool, layout cache) do not apply to the combined pool.
\tif llamaServerRPCServers() != "" {
\t\tos.Unsetenv("OLLAMA_MAIN_GPU_IDX")
\t\treturn
\t}
'''

HELPER_FUNC = '''
// [OLLAMA_RPC_SERVERS patch]
// llamaServerRPCServers returns the normalised OLLAMA_RPC_SERVERS list
// (host:port, comma separated) or "" if the cross-host pool is off.
// Entries without a port are dropped with a warning: llama.cpp would abort the
// whole model load on them.
func llamaServerRPCServers() string {
	raw := strings.TrimSpace(os.Getenv("OLLAMA_RPC_SERVERS"))
	if raw == "" {
		return ""
	}
	var servers []string
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		idx := strings.LastIndex(entry, ":")
		if idx <= 0 || idx == len(entry)-1 {
			slog.Warn("OLLAMA_RPC_SERVERS: ignoring entry without host:port", "entry", entry)
			continue
		}
		if _, err := strconv.Atoi(entry[idx+1:]); err != nil {
			slog.Warn("OLLAMA_RPC_SERVERS: ignoring entry with invalid port", "entry", entry)
			continue
		}
		servers = append(servers, entry)
	}
	return strings.Join(servers, ",")
}
'''


def find_target(ollama_root: Path) -> Path | None:
    candidates = [ollama_root / TARGET_FILE, ollama_root / "src" / TARGET_FILE]
    for c in candidates:
        if c.is_file():
            return c
    return None


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    if "func selectGPUPool(" not in content:
        print("  selectGPUPool() missing: run patch-ollama-dynamic-pool.py first", file=sys.stderr)
        return False
    for label, anchor in (("launch", LAUNCH_ANCHOR), ("pool", POOL_ANCHOR)):
        if content.count(anchor) != 1:
            print(f"  {label} anchor not found exactly once: {anchor!r}", file=sys.stderr)
            return False

    content = content.replace(LAUNCH_ANCHOR, LAUNCH_CODE + LAUNCH_ANCHOR, 1)
    content = content.replace(POOL_ANCHOR, POOL_CODE + POOL_ANCHOR, 1)
    content = content.rstrip("\n") + "\n" + HELPER_FUNC

    path.write_text(content)
    print(f"  RPC pool patch applied to {path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"Looking for {TARGET_FILE} under {root}...")
    target = find_target(root)
    if not target:
        sys.exit(f"Required patch target missing: {TARGET_FILE}")
    print(f"  Target: {target}")
    if not patch(target):
        sys.exit("RPC pool patch failed; review upstream changes before releasing")


if __name__ == "__main__":
    main()
