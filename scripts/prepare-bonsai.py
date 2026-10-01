#!/usr/bin/env python3
"""Prepare the pinned Prism backend and Ollama GGUF support before CMake.

Only used by BONSAI=ON builds. Native libraries must all be built from this
source: Prism's private GGML types must never be mixed with upstream libraries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def apply_patch(root, patch):
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(root.parent))
    base = ["git", "-C", str(root), "apply"]
    if subprocess.run(base + ["--reverse", "--check", str(patch)], env=env,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    subprocess.run(base + ["--check", str(patch)], env=env, check=True)
    subprocess.run(base + [str(patch)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--patches", type=Path,
                        default=Path(__file__).resolve().parent.parent / "patches/bonsai")
    parser.add_argument("--cuda-version", choices=("11", "12", "13"), default="12",
                        help="Ollama CUDA runner variant being built")
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded, checksum-verified archive")
    args = parser.parse_args()
    root, patches = args.root.resolve(), args.patches.resolve()
    source = json.loads((patches / "source.json").read_text())
    target = root / f"build/llama-server-cuda_v{args.cuda_version}/_deps/llama_cpp-src"
    marker = target / ".ollama-bonsai-source.json"
    if target.exists() and not marker.exists():
        raise SystemExit(f"Refusing to overlay an unverified backend: {target}; use a clean build directory")
    if marker.exists() and json.loads(marker.read_text()) != source:
        raise SystemExit("Bonsai backend pin changed; use a clean build directory")

    # Check importer compatibility before downloading the backend.
    apply_patch(root, patches / "ollama-gguf.patch")
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="ollama-bonsai-") as tmp:
            archive = args.archive or Path(tmp) / "source.tar.gz"
            if not args.archive:
                subprocess.run(["curl", "-fL", "--retry", "3", "--output", str(archive),
                                f"https://codeload.github.com/PrismML-Eng/llama.cpp/tar.gz/{source['commit']}"], check=True)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != source["archive_sha256"]:
                raise SystemExit(f"Prism archive checksum mismatch: {digest}")
            extracted = Path(tmp) / "extracted"
            extracted.mkdir()
            with tarfile.open(archive) as tar:
                # No links or paths outside this verified archive's root.
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        raise SystemExit(f"Unexpected archive link: {member.name}")
                    if not (extracted / member.name).resolve().is_relative_to(extracted):
                        raise SystemExit(f"Unsafe archive path: {member.name}")
                tar.extractall(extracted)
            children = list(extracted.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise SystemExit("Unexpected Prism archive layout")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(children[0]), target)
            marker.write_text(json.dumps(source, indent=2) + "\n")

    # Preserve Ollama's compatibility layer, adapting its call sites to the
    # pinned Prism loader rather than disabling compatibility for other models.
    shutil.copyfile(patches / "001-llama-cpp-hooks.patch",
                    root / "llama/compat/001-llama-cpp-hooks.patch")
    shutil.copyfile(patches / "bonsai_test.go", root / "fs/gguf/bonsai_test.go")
    (root / "BONSAI_SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"Prepared {source['release']} at {target}")


if __name__ == "__main__":
    main()
