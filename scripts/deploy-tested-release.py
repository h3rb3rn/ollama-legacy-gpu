#!/usr/bin/env python3
"""Replace one explicitly managed container with its tested image, or roll back.

Only the simple bind-mounted, bridge-network Ollama deployment is supported.
Unrecognized configuration fails before stopping anything. Keep the stopped
previous container for recovery; never delete model data or other instances.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate', ROOT / 'scripts/test-release-gpu.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def post(url, body, timeout=600):
    request = urllib.request.Request(url + '/api/generate', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_old(row, config):
    deployment = config['deployment']
    host = row['HostConfig']
    if (host['NetworkMode'] not in ('bridge', 'default') or host.get('Privileged') or host.get('AutoRemove')
            or host.get('CapAdd') or host.get('Devices') or host.get('SecurityOpt')
            or host.get('VolumesFrom') or host.get('Links')):
        raise ValueError('Deployment has custom isolation settings; preserve them in a dedicated adapter')
    mounts = row['Mounts']
    if (len(mounts) != 1 or mounts[0]['Type'] != 'bind'
            or mounts[0]['Source'] != config['model_root']
            or mounts[0]['Destination'] != '/root/.ollama' or not mounts[0]['RW']):
        raise ValueError('Production model mount differs from the validated shared pool')
    requests = host['DeviceRequests']
    if len(requests) != 1 or set(requests[0].get('DeviceIDs', [])) != set(config['gpu_uuids']):
        raise ValueError('Production GPU UUIDs differ from the validated pool')
    ports = host['PortBindings']
    if ports != {'11434/tcp': [{'HostIp': deployment['bind_ip'], 'HostPort': str(deployment['port'])}]}:
        raise ValueError('Production port mapping differs from the explicit deployment target')
    if row['Config']['Cmd'] != ['serve'] or row['Config'].get('User'):
        raise ValueError('Custom command/user needs a dedicated deployment adapter')
    if not row['State']['Running']:
        raise ValueError('Production must be running before replacement')


def replace(old, name, new_id, backup, temporary, verify, report):
    stopped = renamed = False
    restart = old['HostConfig']['RestartPolicy']
    restart_arg = restart['Name']
    if restart_arg == 'on-failure' and restart['MaximumRetryCount']:
        restart_arg += ':' + str(restart['MaximumRetryCount'])
    try:
        if candidate.inspect(name)['Id'] != old['Id']:
            raise RuntimeError('Production changed after preflight')
        candidate.run('docker', 'stop', '-t', '30', old['Id'])
        stopped = True
        candidate.run('docker', 'rename', old['Id'], backup)
        renamed = True
        candidate.run('docker', 'rename', new_id, name)
        candidate.run('docker', 'start', new_id)
        verify()
        candidate.run('docker', 'update', '--restart=no', old['Id'])
        report.update(passed=True, container_id=new_id)
    except BaseException as exc:
        report.update(passed=False, error=str(exc))
        try:
            candidate.run('docker', 'stop', '-t', '10', new_id)
            candidate.run('docker', 'rename', new_id, temporary + '-failed')
            if renamed:
                candidate.run('docker', 'rename', old['Id'], name)
            if stopped:
                candidate.run('docker', 'update', '--restart=' + restart_arg, old['Id'])
                candidate.run('docker', 'start', old['Id'])
                candidate.wait_api(old['Id'])
            report['rolled_back'] = stopped
        except Exception as rollback_error:
            report['rollback_error'] = str(rollback_error)
            raise RuntimeError(f'{exc}; rollback also failed: {rollback_error}') from exc
        raise


def create_replacement(old, image, config, temporary):
    deployment = config['deployment']
    # Copy every HostConfig field through Docker's API, preserving resource limits,
    # restart policy, log options, device requests and bind/port configuration.
    # Docker CLI has no lossless inspect->create operation.
    import http.client
    import socket

    class DockerConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect('/var/run/docker.sock')

    new = dict(old['Config'])
    image_config = candidate.inspect(image)['Config']
    new['Image'] = image
    new['Hostname'] = ''
    new['Entrypoint'] = image_config['Entrypoint']
    new['Healthcheck'] = image_config.get('Healthcheck')
    env = dict(item.split('=', 1) for item in new['Env'])
    defaults = dict(item.split('=', 1) for item in image_config['Env'])
    env['LD_LIBRARY_PATH'] = defaults['LD_LIBRARY_PATH']
    env.update({'OLLAMA_FLASH_ATTENTION': 'false' if deployment['kv_type'] == 'f16' else 'true',
                'OLLAMA_HOST': '0.0.0.0:11434', 'OLLAMA_INTERNAL_PORT': '11434',
                'OLLAMA_KV_CACHE_TYPE': deployment['kv_type'], 'OLLAMA_NUM_PARALLEL': '1',
                'OLLAMA_SCHED_SPREAD': 'true', 'OLLAMA_CONTEXT_LENGTH': str(config['context']),
                'OLLAMA_AUTO_OPTIMIZE': '0', 'OLLAMA_GPU_AUTODETECT': '0',
                'CUDA_VISIBLE_DEVICES': ','.join(config['gpu_uuids'])})
    for key in ('OLLAMA_FAST_GPU_DEVICES', 'OLLAMA_FAST_POOL_VRAM_GB',
                'OLLAMA_FORCE_GPU_LAYERS', 'OLLAMA_CACHED_TENSOR_SPLIT'):
        env[key] = ''
    new['Env'] = [key + '=' + value for key, value in env.items()]
    new['HostConfig'] = old['HostConfig']
    conn = DockerConnection('localhost', timeout=60)
    from urllib.parse import urlencode
    conn.request('POST', '/containers/create?' + urlencode({'name': temporary}),
                 json.dumps(new), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    if response.status != 201:
        raise RuntimeError(f'Candidate creation failed: {body.get("message")}')
    return body['Id']



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = candidate.load_config(args.config)
    evidence = json.loads(args.evidence.read_text())
    if (not evidence['passed'] or evidence['local_image'] or evidence['target'] != config['id']
            or evidence['config_sha256'] != hashlib.sha256(args.config.read_bytes()).hexdigest()
            or evidence['variant'] != config['variant']):
        raise ValueError('Successful release evidence for this exact target configuration is required')
    image = evidence['image']
    if not re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('Deployment requires the tested immutable registry digest')
    deployment = config['deployment']
    name = deployment['container']
    if name not in config['protected_containers']:
        raise ValueError('Deployment target was not protected during hardware validation')
    if deployment['kv_type'] not in {check['cache'] for check in evidence['checks']}:
        raise ValueError('Requested production KV type was not tested')
    old = candidate.inspect(name)
    validate_old(old, config)
    protected = {other: candidate.identity(other) for other in config['protected_containers'] if other != name}
    candidate.run('docker', 'pull', image, timeout=1800)
    if candidate.inspect(image)['Id'] != evidence['image_id']:
        raise ValueError('Pulled image differs from tested image')
    url = f'http://127.0.0.1:{deployment["port"]}'
    with urllib.request.urlopen(url + '/api/ps', timeout=10) as response:
        if json.load(response)['models']:
            raise RuntimeError('Production has loaded models; retry during the configured idle window')
    suffix = uuid.uuid4().hex[:10]
    backup = name + '-previous-' + suffix
    temporary = name + '-release-' + suffix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'target': config['id'], 'image': image,
              'previous_container': old['Id'], 'rollback_container': backup}
    new_id = create_replacement(old, image, config, temporary)

    def verify():
        candidate.wait_api(new_id)
        result = post(url, {'model': config['bonsai_model'], 'stream': False,
                            'prompt': 'Compute 19 * 23. Answer only with the number.',
                            'keep_alive': 0, 'options': {'num_ctx': config['context'],
                            'num_batch': 128, 'num_predict': 512, 'temperature': 0, 'seed': 42}})
        final = result.get('response', '').rsplit('</think>', 1)[-1].strip()
        if result.get('done_reason') != 'stop' or not re.fullmatch(r'(?:\\boxed\{)?437\}?[.!]?', final):
            raise RuntimeError('Post-deployment inference failed')
        spec = importlib.util.spec_from_file_location('validator', ROOT / 'scripts/validate-bonsai-gpu.py')
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        logs = subprocess.run(['docker', 'logs', new_id], check=True, capture_output=True, text=True)
        validator.validate_logs(logs.stdout + logs.stderr, deployment['kv_type'], config['context'])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = candidate.inspect(new_id)
            if not state['State']['Running'] or state['RestartCount']:
                raise RuntimeError('New production instance exited or restarted')
            if state['State'].get('Health', {}).get('Status') == 'healthy':
                break
            time.sleep(1)
        else:
            raise TimeoutError('New production instance did not become healthy')
        if {other: candidate.identity(other) for other in protected} != protected:
            raise RuntimeError('Another protected instance changed')
        report['version'] = candidate.wait_api(new_id)[1]

    def interrupted(signum, frame):
        raise RuntimeError(f'Deployment interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        replace(old, name, new_id, backup, temporary, verify, report)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
