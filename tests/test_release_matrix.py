import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matrix', ROOT / 'scripts/release-matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class ToolkitRouting(unittest.TestCase):
    def setUp(self):
        self.rows = matrix.targets(ROOT / 'presets/gpu-targets.json')

    def test_k80_retains_cuda11(self):
        self.assertEqual(matrix.select(self.rows, [37, 37]), 'cuda11-legacy')

    def test_maxwell_and_pascal_require_cuda12(self):
        self.assertEqual(matrix.select(self.rows, [50, 52, 61]), 'cuda12-maxwell')

    def test_turing_ampere_rtx_pool(self):
        self.assertEqual(matrix.select(self.rows, [75, 86]), 'cuda13-rtx')

    def test_blackwell(self):
        self.assertEqual(matrix.select(self.rows, [120]), 'cuda13-rtx')

    def test_unknown_gpu_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            matrix.select(self.rows, [999])

    def test_mixed_k80_rtx_requires_separate_pools(self):
        with self.assertRaises(ValueError):
            matrix.select(self.rows, [37, 86])


if __name__ == '__main__':
    unittest.main()
