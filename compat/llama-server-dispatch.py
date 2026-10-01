#!/usr/bin/env python3
"""Narrow Maxwell compatibility dispatch; never load Prism tensors upstream.

The pinned Prism backend lacks spark2_5. Its known-working, separately bundled
Maxwell backend handles only ordinary spark2_5 GGUFs. Bonsai and discovery use
Prism. Each process loads one complete backend with its private libraries.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path('/usr/lib/ollama')
# Outside the primary discovery tree: Ollama enumerates backend subdirectories
# beneath /usr/lib/ollama. Nesting the fallback there leaks its CUDA library into
# Prism discovery even when the selected process uses primary LD_LIBRARY_PATH.
COMPAT = Path('/opt/ollama-spark-compat')
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}

def metadata(path):
    """Read only enough bounded GGUF metadata to select the backend."""
    with Path(path).open('rb') as file:
        size = os.fstat(file.fileno()).st_size
        def read(n):
            if n < 0 or file.tell() + n > size or file.tell() + n > 32 * 1024 * 1024:
                raise ValueError('Truncated or oversized GGUF metadata')
            data = file.read(n)
            if len(data) != n:
                raise ValueError('Truncated GGUF')
            return data
        def u32():
            return struct.unpack('<I', read(4))[0]
        def u64():
            return struct.unpack('<Q', read(8))[0]
        def string():
            n = u64()
            if n > 1024 * 1024:
                raise ValueError('Oversized GGUF string')
            return read(n).decode('utf-8')
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = u32(), u64()
                if count > 1000000 or element == 9:
                    raise ValueError('Unsupported GGUF array')
                if element in SCALAR_BYTES:
                    read(count * SCALAR_BYTES[element])
                else:
                    for _ in range(count):
                        value(element)
                return None
            if kind not in SCALAR_BYTES:
                raise ValueError('Unknown GGUF metadata type')
            data = read(SCALAR_BYTES[kind])
            return int.from_bytes(data, 'little')
        if read(4) != b'GGUF' or u32() not in (2, 3):
            raise ValueError('Unsupported GGUF header')
        tensors = u64()
        count = u64()
        if count > 100000:
            raise ValueError('Oversized GGUF metadata table')
        found = {}
        for _ in range(count):
            key, kind = string(), u32()
            item = value(kind)
            if key in ('general.architecture', 'general.file_type'):
                found[key] = item
            if len(found) == 2 and not uses_compat(found):
                return found
        if uses_compat(found):
            if tensors > 100000:
                raise ValueError('Oversized GGUF tensor table')
            for _ in range(tensors):
                string()  # tensor name
                dimensions = u32()
                if not 1 <= dimensions <= 4:
                    raise ValueError('Invalid GGUF tensor dimensions')
                read(8 * dimensions)
                tensor_type = u32()
                u64()  # data offset
                if tensor_type >= 140:
                    found['private_tensor_types'] = True
        return found

def uses_compat(info):
    # Private Prism file types140+ MUST stay with Prism, even if an altered
    # architecture string claims spark2_5. Only known ordinary types qualify.
    return (info.get('general.architecture') == 'spark2_5'
            and not info.get('private_tensor_types')
            and isinstance(info.get('general.file_type'), int)
            and 0 <= info['general.file_type'] <= 39)

def maxwell_devices():
    """The compatibility artifact contains sm50/sm52, never other GPUs."""
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,compute_cap',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = [tuple(x.strip() for x in line.split(',')) for line in output.splitlines()]
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if selected:
        ids = selected.split(',')
        available = dict(rows)
        capabilities = [available.get(gpu) if gpu.startswith('GPU-')
                        else rows[int(gpu)][1] if gpu.isdigit() and int(gpu) < len(rows) else None
                        for gpu in ids]
    else:
        capabilities = [row[1] for row in rows]
    return bool(capabilities) and all(cap in ('5.0', '5.2') for cap in capabilities)

def main():
    args = sys.argv[1:]
    model = None
    for i, arg in enumerate(args):
        if arg in ('--model', '-m') and i + 1 < len(args):
            model = args[i + 1]
        elif arg.startswith('--model='):
            model = arg.split('=', 1)[1]
    info = {}
    if model:
        try:
            info = metadata(model)
        except (OSError, ValueError, UnicodeError, struct.error) as exc:
            print('backend-dispatch: metadata unavailable; retaining Prism: ' + str(exc), file=sys.stderr)
    env = os.environ.copy()
    if uses_compat(info):
        if not maxwell_devices():
            raise RuntimeError('Pinned Spark compatibility backend is limited to Maxwell sm50/sm52')
        binary = COMPAT / 'llama-server'
        # Remove primary library paths, so the private GGML enum spaces and
        # native implementation libraries cannot accidentally be combined.
        inherited = [p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                     if p and not (p == str(ROOT) or p.startswith(str(ROOT) + '/')
                                   or p == str(COMPAT) or p.startswith(str(COMPAT) + '/'))]
        env['LD_LIBRARY_PATH'] = ':'.join([str(COMPAT / 'cuda_v12'), str(COMPAT), *inherited])
        env['GGML_BACKEND_PATH'] = str(COMPAT)
        print('backend-dispatch: spark2_5 -> isolated pinned Maxwell compatibility backend', file=sys.stderr)
    else:
        binary = ROOT / 'llama-server-bonsai'
        env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                                         if p and not (p == str(COMPAT) or p.startswith(str(COMPAT) + '/')))
        env['GGML_BACKEND_PATH'] = str(ROOT)
        if model:
            print('backend-dispatch: model -> pinned Prism Bonsai backend', file=sys.stderr)
    os.execve(str(binary), [str(binary), *args], env)

if __name__ == '__main__':
    main()
