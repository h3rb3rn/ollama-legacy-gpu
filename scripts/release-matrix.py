#!/usr/bin/env python3
"""Emit release build targets; reject unsupported GPU/toolkit combinations."""
import argparse
import json
from pathlib import Path


def targets(path):
    rows = json.loads(path.read_text())
    for variant, row in rows.items():
        caps = row['compute_capabilities']
        if not caps or len(set(caps)) != len(caps):
            raise ValueError(f'{variant}: empty/duplicate GPU architectures')
        if variant.startswith('cuda13') and min(caps) < 75:
            raise ValueError('CUDA 13 cannot target Kepler/Maxwell/Pascal/Volta')
        if variant.startswith('cuda12') and min(caps) < 50:
            raise ValueError('CUDA 12 cannot target K80')
        real = {int(a.removesuffix('-real')) for a in row['architectures'].split(';')
                if a.endswith('-real')}
        if set(caps) != real:
            raise ValueError(f'{variant}: advertised GPUs differ from real build targets')
        if row['flash_attention'] != 'ON' and row['validation_cache_types'] != ['f16']:
            raise ValueError(f'{variant}: quantized KV requires compiled Flash Attention')
    return rows


def select(rows, caps):
    """Prefer the newest toolkit that can compile every requested GPU."""
    if not caps:
        raise ValueError('No GPU compute capabilities supplied')
    for variant in reversed(sorted(rows)):
        if set(caps) <= set(rows[variant]['compute_capabilities']):
            return variant
    raise ValueError('No common build target; split these GPUs into compatible pools')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'presets/gpu-targets.json')
    parser.add_argument('--compute-capabilities', type=int, nargs='+')
    args = parser.parse_args()
    rows = targets(args.targets)
    if args.compute_capabilities is not None:
        print(select(rows, args.compute_capabilities))
    else:
        print(json.dumps({'include': [{'variant': key, **value} for key, value in rows.items()]}))


if __name__ == '__main__':
    main()
