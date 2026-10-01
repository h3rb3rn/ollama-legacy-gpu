#!/usr/bin/env python3
"""Test an immutable candidate on an explicitly configured GPU pool.

Runs on the GPU host, using its existing model directory. Does not stop or
replace production. A busy pool fails closed; the workflow may retry later.
Configuration is a local JSON file, selected by the release inventory.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs).stdout


def inspect(name):
    return json.loads(run('docker', 'inspect', name))[0]


def identity(name):
    row = inspect(name)
    return {key: value for key, value in (
        ('id', row['Id']), ('image', row['Image']), ('started', row['State']['StartedAt']),
        ('restarts', row['RestartCount']), ('running', row['State']['Running']),
    )}


def load_config(path):
    config = json.loads(path.read_text())
    if config['family'] not in ('k80', 'm10', 'm60', 'rtx'):
        raise ValueError('Unknown hardware family')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', config['id']):
        raise ValueError('Invalid target identifier')
    gpus = config['gpu_uuids']
    if not gpus or len(set(gpus)) != len(gpus) or any(
            not re.fullmatch(r'GPU-[0-9a-f-]{36}', gpu) for gpu in gpus):
        raise ValueError('Explicit unique GPU UUIDs are required')
    if not Path(config['model_root']).is_absolute():
        raise ValueError('An absolute shared model directory is required')
    if not config['protected_containers']:
        raise ValueError('Name the production containers whose identity must remain unchanged')
    if config['context'] < 4096:
        raise ValueError('Context must be at least 4096')
    if config['family'] in ('m10', 'm60') and config['context'] != 190000:
        raise ValueError('Maxwell Bonsai validation requires the approved 190000 context')
    if config['family'] == 'rtx' and config['context'] != 262144:
        raise ValueError('RTX Bonsai validation requires 262144 context')
    if config['bonsai_model'] == config['ordinary_model']:
        raise ValueError('Ordinary GGUF regression requires a separate model')
    return config


def available_gpus(config):
    result = run('nvidia-smi', '--query-gpu=uuid,name,compute_cap,memory.used',
                 '--format=csv,noheader,nounits')
    all_gpus = {}
    for line in result.splitlines():
        gpu, name, capability, used = [item.strip() for item in line.split(',')]
        all_gpus[gpu] = {'name': name, 'capability': int(round(float(capability) * 10)),
                         'used_mib': int(used)}
    selected = {gpu: all_gpus[gpu] for gpu in config['gpu_uuids']}
    expected = {'k80': 37, 'm10': 50, 'm60': 52}.get(config['family'])
    for gpu, data in selected.items():
        if expected and data['capability'] != expected:
            raise RuntimeError(f'{gpu} is not the configured hardware family')
        if config['family'] == 'rtx' and data['capability'] < 75:
            raise RuntimeError('CUDA 13 requires Turing or newer')
        if data['used_mib'] > 128:
            raise RuntimeError(f'{gpu} is busy ({data["used_mib"]} MiB); no production unload attempted')
    return selected


def wait_api(name):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        row = inspect(name)
        if not row['State']['Running'] or row['RestartCount']:
            raise RuntimeError('Candidate exited or restarted during startup')
        ports = row['NetworkSettings']['Ports'].get('11434/tcp')
        if not ports:
            time.sleep(1)
            continue
        url = 'http://127.0.0.1:' + ports[0]['HostPort']
        try:
            with urllib.request.urlopen(url + '/api/version', timeout=2) as response:
                return url, json.load(response)
        except (OSError, ValueError):
            time.sleep(1)
    raise TimeoutError('Candidate did not become ready in 120 seconds')


def candidate_args(config, image, name, cache):
    env = {
        'OLLAMA_HOST': '0.0.0.0:11434', 'OLLAMA_INTERNAL_PORT': '11434',
        'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_SCHED_SPREAD': 'true',
        'OLLAMA_FLASH_ATTENTION': 'false' if cache == 'f16' else 'true',
        'OLLAMA_KV_CACHE_TYPE': cache, 'OLLAMA_LOAD_TIMEOUT': '30m',
        'OLLAMA_CONTEXT_LENGTH': str(config['context']),
        'OLLAMA_AUTO_OPTIMIZE': '0', 'OLLAMA_GPU_AUTODETECT': '0',
        'OLLAMA_FAST_GPU_DEVICES': '', 'OLLAMA_FAST_POOL_VRAM_GB': '',
        'OLLAMA_FORCE_GPU_LAYERS': '', 'OLLAMA_CACHED_TENSOR_SPLIT': '',
        'CUDA_VISIBLE_DEVICES': ','.join(config['gpu_uuids']),
    }
    args = ['docker', 'create', '--name', name, '--restart', 'no',
            '--label', 'ollama.release-candidate=true',
            '--gpus', '"device=' + ','.join(config['gpu_uuids']) + '"',
            '-p', '127.0.0.1::11434',
            '--mount', f'type=bind,source={config["model_root"]},target=/root/.ollama']
    for key, value in env.items():
        args += ['-e', key + '=' + value]
    return args + [image, 'serve']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-image', action='store_true',
                        help='Use a locally built image ID; never accepted by release promotion')
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.local_image and not re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}', args.image):
        parser.error('Release tests require an immutable GHCR digest')
    args.output.mkdir(parents=True, exist_ok=True)
    # Per-GPU locks also serialize partially overlapping configured pools.
    locks = []
    for gpu in sorted(config['gpu_uuids']):
        lock = open('/tmp/ollama-validation-' + gpu + '.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    protected = {name: identity(name) for name in config['protected_containers']}
    if not all(row['running'] for row in protected.values()):
        raise RuntimeError('A protected production container is already stopped')
    hardware = available_gpus(config)
    spec = importlib.util.spec_from_file_location('release_matrix', ROOT / 'scripts/release-matrix.py')
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    targets = matrix.targets(ROOT / 'presets/gpu-targets.json')
    variant = matrix.select(targets, [row['capability'] for row in hardware.values()])
    if variant != config['variant']:
        raise RuntimeError(f'Target requires {variant}, not {config["variant"]}')
    if not args.local_image:
        run('docker', 'pull', args.image, timeout=1800)
    image_id = inspect(args.image)['Id']
    report = {'passed': False, 'target': config['id'], 'family': config['family'],
              'variant': variant, 'image': args.image, 'image_id': image_id,
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'local_image': args.local_image, 'hardware': hardware, 'checks': []}
    try:
        for cache in targets[variant]['validation_cache_types']:
            available_gpus(config)
            name = 'ollama-test-' + config['id'] + '-' + uuid.uuid4().hex[:10]
            candidate_id = None
            try:
                candidate_id = run(*candidate_args(config, image_id, name, cache)).strip()
                run('docker', 'start', name)
                url, version = wait_api(name)
                report['version'] = version
                for kind, model, context in [
                    ('bonsai', config['bonsai_model'], config['context']),
                    ('ordinary', config['ordinary_model'], 4096),
                ]:
                    output = args.output / f'{cache}-{kind}.json'
                    subprocess.run([sys.executable, str(ROOT / 'scripts/validate-bonsai-gpu.py'),
                                    '--url', url, '--container', name, '--model', model,
                                    '--kv-type', cache, '--context', str(context),
                                    '--output', str(output)], check=True, timeout=7200)
                    evidence = json.loads(output.read_text())
                    if not evidence['passed'] or evidence['container']['image_id'] != image_id:
                        raise RuntimeError('Candidate evidence does not match the requested image')
                    report['checks'].append({'cache': cache, 'kind': kind, 'evidence': output.name})
            finally:
                if candidate_id:
                    # Use ID, so even an unexpected rename cannot remove another instance.
                    candidate = inspect(candidate_id)
                    if candidate['Config']['Labels'].get('ollama.release-candidate') != 'true':
                        raise RuntimeError('Candidate identity changed; refusing cleanup')
                    logs = subprocess.run(['docker', 'logs', candidate['Id']], capture_output=True, text=True)
                    (args.output / f'{cache}-container.log').write_text(logs.stdout + logs.stderr)
                    run('docker', 'rm', '-f', candidate['Id'])
        if {name: identity(name) for name in protected} != protected:
            raise RuntimeError('Production identity/restart count changed during validation')
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
