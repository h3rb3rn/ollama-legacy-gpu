"""Regression checks for evidence that must prevent GPU build promotion."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'validate_gpu', Path(__file__).resolve().parents[1] / 'scripts/validate-bonsai-gpu.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GPUValidation(unittest.TestCase):
    valid = '''load_tensors: offloaded 65/65 layers to GPU
llama_kv_cache: size = 3343.50 MiB, K (q4_0): 1671.75 MiB, V (q4_0): 1671.75 MiB
'''

    def test_full_gpu_q4(self):
        self.assertEqual(module.validate_logs(self.valid, 'q4_0'), [('65', '65')])

    def test_full_gpu_f16_legacy(self):
        self.assertEqual(module.validate_logs(self.valid.replace('q4_0', 'f16'), 'f16'),
                         [('65', '65')])

    def test_full_gpu_q8(self):
        self.assertEqual(module.validate_logs(self.valid.replace('q4_0', 'q8_0'), 'q8_0'),
                         [('65', '65')])

    def test_partial_cpu_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs(self.valid.replace('65/65', '64/65'), 'q4_0')

    def test_cache_fallback_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs(self.valid.replace('V (q4_0)', 'V (f16)'), 'q4_0')

    def test_wrong_requested_type_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs(self.valid, 'q8_0')

    def test_recovered_oom_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs('cudaMalloc failed: out of memory\n' + self.valid, 'q4_0')

    def test_missing_evidence_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs('server healthy', 'q4_0')

    def test_truncated_stability_run_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_generation({'done': True, 'eval_count': 128,
                                        'eval_duration': 1000000000}, 2048)

    def test_success_flag_does_not_hide_api_error(self):
        with self.assertRaises(RuntimeError):
            module.validate_generation({'done': True, 'error': 'CUDA error',
                                        'eval_count': 2048, 'eval_duration': 1000000000})

    def test_timings_use_nanoseconds(self):
        row = module.measure({'context': [1, 2, 3], 'eval_count': 128,
                              'eval_duration': 8000000000})
        self.assertEqual(row['eval_tps'], 16)
        self.assertNotIn('context', row)

    def test_requested_large_context_cannot_silently_shrink(self):
        log = self.valid + 'llama_context: n_ctx = 4096\n'
        with self.assertRaises(RuntimeError):
            module.validate_logs(log, 'q4_0', 190000)

    def test_expected_padding_is_accepted(self):
        log = self.valid + 'llama_context: n_ctx = 190208\n'
        self.assertTrue(module.validate_logs(log, 'q4_0', 190000))

    def test_mixed_cache_types_are_rejected(self):
        with self.assertRaises(RuntimeError):
            module.validate_logs(self.valid + self.valid.replace('q4_0', 'f16'), 'q4_0')


if __name__ == '__main__':
    unittest.main()
