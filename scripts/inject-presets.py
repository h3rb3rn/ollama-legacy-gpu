#!/usr/bin/env python3
"""inject-presets.py — Injects custom CMake presets into llama/server/CMakePresets.json.

Used to add CUDA 11 (CC 3.7/K80) support to the current Ollama build system,
which dropped CUDA 11 from official presets after v0.30.x.

Usage:
  python3 scripts/inject-presets.py \
    --source  <ollama-source-dir> \
    --presets presets/cuda11-presets.json

Exits 0 on success, 1 on error.
"""

import json
import sys
import argparse
from pathlib import Path


def inject(source_dir: Path, presets_file: Path) -> None:
    target = source_dir / "llama" / "server" / "CMakePresets.json"

    if not target.exists():
        sys.exit(f"ERROR: {target} not found — is this a valid Ollama source tree? "
                 "Expected llama/server/CMakePresets.json (Ollama >= 0.30.x)")

    with open(target) as f:
        existing = json.load(f)

    with open(presets_file) as f:
        additions = json.load(f)

    existing_names = {p["name"] for p in existing.get("configurePresets", [])}
    existing_build_names = {p["name"] for p in existing.get("buildPresets", [])}

    added = []
    for preset in additions.get("configurePresets", []):
        if preset["name"] not in existing_names:
            existing.setdefault("configurePresets", []).append(preset)
            added.append(preset["name"])
        else:
            print(f"  SKIP (already exists): {preset['name']}")

    for preset in additions.get("buildPresets", []):
        if preset["name"] not in existing_build_names:
            existing.setdefault("buildPresets", []).append(preset)
        else:
            print(f"  SKIP build preset (already exists): {preset['name']}")

    with open(target, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")

    if added:
        print(f"Injected presets: {', '.join(added)}")
        print(f"Written: {target}")
    else:
        print("No new presets to inject — all already present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True, help="Ollama source root directory")
    parser.add_argument("--presets", required=True, help="JSON file with presets to inject")
    args = parser.parse_args()
    inject(Path(args.source), Path(args.presets))
