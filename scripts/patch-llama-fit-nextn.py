#!/usr/bin/env python3
"""
patch-llama-fit-nextn.py — makes common/fit.cpp count the nextn (MTP) layer slot even when
MTP is not loaded (RTX-TESLA-GAPS-PROMPT.md, Gap 4).

The loader (llama_model::load_tensors) always lays GPU layers out over n_layer_all + 1
slots (trunk + nextn + output) and offloads the LAST n_gpu_layers of them. The fit code
plans with hp_ngl + 1 slots, where hp_ngl = n_layer, plus n_layer_nextn only if
mparams->load_mtp. With MTP off (our default) the fit therefore hands the loader
n_gpu_layers = 41 for a 42-slot model, i_gpu_start becomes 1 instead of 0, and the FIRST
trunk layer stays on the CPU ("offloaded 41/42", +~520 MiB CPU buffer for qwen35moe).
That layer then runs on the host for every token. The nextn slot itself is empty
when MTP is off, so counting it costs no VRAM.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [FIT_NEXTN_SLOT patch]"
SOURCE_FILE = Path("common") / "fit.cpp"

OLD = '''    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
'''
NEW = '''    hp_ngl         = llama_model_n_layer(model);
    ''' + PATCH_GUARD + ''' the loader always lays out n_layer_all + 1 slots, so the fit
    // must count the nextn slot even when MTP is not loaded; otherwise trunk layer 0 is
    // pushed to the CPU.
    hp_ngl        += llama_model_n_layer_nextn(model);
'''


def find_sources(root: Path):
    out = []
    for variant in ("cuda_v12", "cuda_v13", "cuda_v11"):
        base = root / "build" / f"llama-server-{variant}" / "_deps" / "llama_cpp-src"
        if (base / SOURCE_FILE).is_file():
            out.append(base / SOURCE_FILE)
    return out


def patch(path: Path) -> bool:
    content = path.read_text()
    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True
    if content.count(OLD) != 1:
        print(f"  ERROR: hp_ngl/load_mtp block not found exactly once in {path}", file=sys.stderr)
        return False
    path.write_text(content.replace(OLD, NEW, 1))
    print(f"  nextn slot fit patch applied to {path}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    paths = find_sources(Path(sys.argv[1]))
    if not paths:
        print("ERROR: common/fit.cpp not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(p) for p in paths) else 1


if __name__ == "__main__":
    sys.exit(main())
