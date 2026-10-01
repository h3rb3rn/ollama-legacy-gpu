#!/usr/bin/env python3
"""Validate a candidate Ollama container before promotion (never restarts it).

The endpoint must be a dedicated candidate. The probe checks actual CUDA FA
support, the requested cache type in runner logs, full GPU layer placement,
two deterministic answers, and container identity/restart count. A successful
API health response alone is not sufficient. Emits a JSON evidence artifact.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True)


def fingerprint(container):
    row = json.loads(docker('inspect', container))[0]
    if not row['State']['Running']:
        raise RuntimeError('Candidate is not running')
    return {key: value for key, value in (
        ('id', row['Id']), ('image_id', row['Image']),
        ('started_at', row['State']['StartedAt']), ('restarts', row['RestartCount']),
        ('gpu_requests', row['HostConfig']['DeviceRequests']),
    )}


def post(base, route, body):
    request = urllib.request.Request(base + route, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def validate_logs(logs, kv_type, context=None):
    layers = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', logs)
    if not layers or any(int(actual) != int(total) for actual, total in layers):
        raise RuntimeError(f'Full GPU layer placement not proven: {layers}')
    for cache in ('K', 'V'):
        observed = set(re.findall(r'\b' + cache + r' \((\w+)\):', logs))
        if observed != {kv_type}:
            raise RuntimeError(f'Requested {cache} cache type {kv_type} not observed')
    if context is not None:
        sizes = [int(size) for size in re.findall(r'llama_context:\s+n_ctx\s+=\s+(\d+)', logs)]
        if not sizes or any(size < context or size >= context + 256 for size in sizes):
            raise RuntimeError(f'Requested context allocation not proven: {sizes}')
    errors = re.findall(r'CUDA error:.*|cudaMalloc failed:.*|failed to allocate.*', logs)
    if errors:
        raise RuntimeError(f'CUDA failures: {errors}')
    return layers


def measure(result):
    """Keep response and timing evidence, without the large token-ID array."""
    row = {key: value for key, value in result.items() if key != 'context'}
    for phase in ('prompt_eval', 'eval'):
        duration = result.get(phase + '_duration', 0)
        row[phase + '_tps'] = result.get(phase + '_count', 0) * 1e9 / duration if duration else None
    return row


def validate_generation(result, minimum_tokens=1):
    if (not result.get('done') or result.get('error')
            or result.get('eval_count', 0) < minimum_tokens
            or result.get('eval_duration', 0) <= 0):
        raise RuntimeError('Incomplete or too short inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--model', default='bonsai2:27b-pq2_0')
    parser.add_argument('--kv-type', choices=['f16', 'q8_0', 'q4_0'], required=True)
    parser.add_argument('--context', type=int, default=190000)
    parser.add_argument('--stability-tokens', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.context < 4096 or args.stability_tokens < 1024:
        parser.error('Release validation requires context >=4096 and >=1024 stability tokens')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip('/')
    original = fingerprint(args.container)
    owns_model = False
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    evidence = {'passed': False, 'started': started, 'container': original,
                'model': args.model, 'context': args.context, 'kv_type': args.kv_type,
                'results': []}
    failure = None
    try:
        with urllib.request.urlopen(base + '/api/ps', timeout=10) as response:
            if json.load(response)['models']:
                raise RuntimeError('Candidate must be idle with no loaded model')
        if args.kv_type != 'f16':
            probe = json.loads(docker('exec', args.container, 'python3',
                                     '/usr/local/bin/check-cuda-flash-attn.py'))
            evidence['cuda_capabilities'] = probe
            if not probe.get('passed'):
                raise RuntimeError('CUDA Flash Attention probe failed')
        for question, answer in [('Compute 19 * 23. Answer only with the number.', '437'),
                                 ('What is 1000 minus 237? Answer only with the number.', '763')]:
            print(json.dumps({'phase': 'arithmetic', 'expected': answer}), flush=True)
            t0 = time.monotonic()
            owns_model = True
            result = post(base, '/api/generate', {
                'model': args.model, 'prompt': question, 'stream': False,
                'keep_alive': '5m', 'options': {
                    'num_ctx': args.context, 'num_batch': 128, 'num_predict': 512,
                    'temperature': 0, 'seed': 42,
                },
            })
            # Some pinned templates expose a closing thinking tag in response.
            final = result.get('response', '').rsplit('</think>', 1)[-1].strip()
            correct = bool(re.fullmatch(r'\s*(?:\\boxed\{)?' + answer + r'(?:\})?[.!]?\s*', final))
            evidence['results'].append({'test': 'arithmetic', 'wall_seconds': time.monotonic() - t0,
                                        'expected': answer, 'correct': correct, **measure(result)})
            validate_generation(result)
            if not result.get('done') or result.get('done_reason') != 'stop' or not correct:
                raise RuntimeError(f'Inference correctness failed: {final!r}')
            print(json.dumps({'phase': 'arithmetic-passed', 'answer': answer}), flush=True)
        for label, count, prompt in [
            ('benchmark-1', 128, 'List the integers from 1 to 200, separated by spaces. Do not explain.'),
            ('benchmark-2', 128, 'List the integers from 1 to 200, separated by spaces. Do not explain.'),
            ('stability', args.stability_tokens,
             'Write a detailed tutorial on sorting algorithms with Python implementations, '
             'correctness proofs and complexity analysis. Cover insertion sort, merge sort, '
             'quicksort and heapsort. Provide at least 4000 words.'),
        ]:
            print(json.dumps({'phase': label, 'tokens': count}), flush=True)
            t0 = time.monotonic()
            result = post(base, '/api/generate', {
                'model': args.model, 'prompt': prompt, 'stream': False, 'keep_alive': '5m',
                'options': {'num_ctx': args.context, 'num_batch': 128, 'num_predict': count,
                            'temperature': 0, 'seed': 42},
            })
            evidence['results'].append({'test': label, 'wall_seconds': time.monotonic() - t0,
                                        **measure(result)})
            validate_generation(result, count)
            print(json.dumps({'phase': label + '-passed', 'eval_tps': measure(result)['eval_tps']}),
                  flush=True)
        logs = subprocess.run(['docker', 'logs', '--since', started, args.container],
                              check=True, capture_output=True, text=True)
        full_log = logs.stdout + logs.stderr
        args.output.with_suffix('.runner.log').write_text(full_log)
        evidence['layers'] = validate_logs(full_log, args.kv_type, args.context)
        if fingerprint(args.container) != original:
            raise RuntimeError('Candidate identity or restart count changed')
    except Exception as exc:
        failure = exc
        evidence['error'] = str(exc)
    finally:
        # Preserve diagnostics on failed loads/correctness checks as well.
        logs = subprocess.run(['docker', 'logs', '--since', started, args.container],
                              capture_output=True, text=True)
        args.output.with_suffix('.runner.log').write_text(logs.stdout + logs.stderr)
        # Only this dedicated candidate is addressed, never other instances.
        if owns_model:
            try:
                post(base, '/api/generate', {'model': args.model, 'keep_alive': 0})
            except Exception as exc:
                evidence['cleanup_error'] = str(exc)
                failure = failure or exc
        evidence['passed'] = failure is None
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    if failure is not None:
        raise failure
    print(json.dumps({'passed': True, 'evidence': str(args.output)}))


if __name__ == '__main__':
    main()
