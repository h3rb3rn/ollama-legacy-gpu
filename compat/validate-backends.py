#!/usr/bin/env python3
"""Validate per-process library isolation and restore the production Spark model."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

MAPS_CODE = """import pathlib,json
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\x00',b' ').decode()
  libs=sorted(set(x.split()[-1] for x in (p/'maps').read_text().splitlines() if 'libggml' in x or 'libllama' in x))
  if libs:out.append(dict(pid=p.name,cmd=cmd[:200],libraries=libs))
 except (OSError,UnicodeError):pass
print(json.dumps(out))"""

def verify_maps(rows):
    for row in rows:
        private = row['cmd'].startswith('/opt/ollama-spark-compat/llama-server ')
        for library in row['libraries']:
            if private:
                allowed = library.startswith('/opt/ollama-spark-compat/')
            else:
                allowed = str(Path(library).parent) in (
                    '/usr/lib/ollama', '/usr/lib/ollama/cuda_v12', '/usr/lib/ollama/cuda_v13')
            if not allowed:
                raise RuntimeError('Native library isolation failed: ' + json.dumps(row))

def validate_maps(container):
    rows = json.loads(subprocess.check_output(['docker', 'exec', container, 'python3', '-c', MAPS_CODE]))
    verify_maps(rows)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--spark-model', default='hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M')
    parser.add_argument('--spark-context', type=int, default=98304)
    parser.add_argument('--prism-model', default='qwen3.5:4b')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = dict(passed=False)

    def fingerprint():
        row = json.loads(subprocess.check_output(['docker', 'inspect', args.container]))[0]
        return dict(id=row['Id'], image=row['Image'], started=row['State']['StartedAt'],
                    running=row['State']['Running'], restarts=row['RestartCount'])

    def generate(model, context, prompt='Reply with exactly: 437'):
        body = dict(model=model, stream=False, think=False, prompt=prompt, keep_alive='24h',
                    options=dict(num_ctx=context, num_batch=64, num_predict=128, temperature=0, seed=42))
        req = urllib.request.Request(args.url.rstrip('/') + '/api/generate', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)

    original = fingerprint()
    evidence['container'] = original
    failure = None
    try:
        result = generate(args.prism_model, 4096)
        assert result.get('done') and '437' in result.get('response', '')
        evidence['prism_response'] = {k: result.get(k) for k in ('model', 'response', 'eval_count', 'eval_duration')}
        evidence['prism_maps'] = validate_maps(args.container)
        assert any(r['cmd'].startswith('/usr/lib/ollama/llama-server-bonsai ') for r in evidence['prism_maps'])
        result = generate(args.spark_model, args.spark_context)
        assert result.get('done') and '437' in result.get('response', '')
        evidence['spark_response'] = {k: result.get(k) for k in ('model', 'response', 'eval_count', 'eval_duration')}
        evidence['spark_maps'] = validate_maps(args.container)
        assert any(r['cmd'].startswith('/opt/ollama-spark-compat/llama-server ') for r in evidence['spark_maps'])
        assert fingerprint() == original
        evidence['passed'] = True
    except BaseException as exc:
        failure = exc
        evidence['error'] = str(exc)
    finally:
        # Preserve the expected production model even if the isolation check fails.
        try:
            result = generate(args.spark_model, args.spark_context, prompt='')
            assert result.get('done')
            evidence['production_restored'] = True
        except BaseException as exc:
            evidence['restoration_error'] = str(exc)
            evidence['passed'] = False
            failure = failure or exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(passed=evidence['passed'], evidence=str(args.output))), flush=True)
    if failure:
        raise failure

if __name__ == '__main__':
    main()
