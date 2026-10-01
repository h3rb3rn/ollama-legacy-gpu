"""Dispatch isolation regressions, including private Prism GGUF file types."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dispatch', Path(__file__).with_name('llama-server-dispatch.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
isolation_spec = importlib.util.spec_from_file_location('isolation', Path(__file__).with_name('validate-backends.py'))
isolation = importlib.util.module_from_spec(isolation_spec)
isolation_spec.loader.exec_module(isolation)

def string(value):
    b = value.encode()
    return struct.pack('<Q', len(b)) + b

class DispatchTests(unittest.TestCase):
    def test_primary_process_cannot_load_nested_fallback_cuda_library(self):
        with self.assertRaises(RuntimeError):
            isolation.verify_maps([dict(cmd='/usr/lib/ollama/llama-server-bonsai --model x',
                                        libraries=['/usr/lib/ollama/libggml-base.so',
                                                   '/usr/lib/ollama/spark-compat/libggml-cuda.so'])])

    def test_concurrent_processes_keep_separate_library_sets(self):
        isolation.verify_maps([
            dict(cmd='/usr/lib/ollama/llama-server-bonsai --model x', libraries=['/usr/lib/ollama/libggml-base.so']),
            dict(cmd='/opt/ollama-spark-compat/llama-server --model y', libraries=['/opt/ollama-spark-compat/libggml-base.so'])])

    def test_maxwell_fallback_rejects_other_gpu_architectures(self):
        with patch.object(d.subprocess, 'check_output', return_value='GPU-m60, 5.2\nGPU-rtx, 8.6\n'):
            with patch.dict(d.os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-m60'}):
                self.assertTrue(d.maxwell_devices())
            with patch.dict(d.os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-rtx'}):
                self.assertFalse(d.maxwell_devices())
            with patch.dict(d.os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-unknown'}):
                self.assertFalse(d.maxwell_devices())

    def test_private_prism_types_never_route_to_ordinary_backend(self):
        for file_type in (140, 141, 142, 143):
            self.assertFalse(d.uses_compat({'general.architecture': 'spark2_5', 'general.file_type': file_type}))

    def test_bonsai_and_unknown_architectures_stay_on_prism(self):
        for architecture in ('qwen35', 'qwen35moe', 'unknown'):
            self.assertFalse(d.uses_compat({'general.architecture': architecture, 'general.file_type': 15}))

    def test_ordinary_spark_metadata_dispatch(self):
        data = struct.pack('<4sIQQ', b'GGUF', 3, 1, 2)
        data += string('general.architecture') + struct.pack('<I', 8) + string('spark2_5')
        data += string('general.file_type') + struct.pack('<II', 4, 15)
        data += string('weight') + struct.pack('<IQQIQ', 2, 32, 32, 12, 0)
        with tempfile.NamedTemporaryFile() as file:
            file.write(data)
            file.flush()
            self.assertTrue(d.uses_compat(d.metadata(file.name)))

    def test_mislabeled_private_tensor_stays_with_prism(self):
        data = struct.pack('<4sIQQ', b'GGUF', 3, 1, 2)
        data += string('general.architecture') + struct.pack('<I', 8) + string('spark2_5')
        data += string('general.file_type') + struct.pack('<II', 4, 15)
        data += string('weight') + struct.pack('<IQQIQ', 2, 32, 32, 142, 0)
        with tempfile.NamedTemporaryFile() as file:
            file.write(data)
            file.flush()
            self.assertFalse(d.uses_compat(d.metadata(file.name)))

    def test_missing_or_truncated_metadata_does_not_enable_fallback(self):
        self.assertFalse(d.uses_compat({'general.architecture': 'spark2_5'}))
        with tempfile.NamedTemporaryFile() as file:
            file.write(b'GGUF')
            file.flush()
            with self.assertRaises(ValueError):
                d.metadata(file.name)

if __name__ == '__main__':
    unittest.main()
