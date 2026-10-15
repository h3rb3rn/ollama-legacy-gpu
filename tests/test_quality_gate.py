import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qg = load('quality-gate')


def pos(token, top):
    """One answer position: the chosen token and its top list as [(token, prob), ...]."""
    return {"token": token, "logprob": math.log(dict(top)[token]),
            "top": [[t, math.log(p)] for t, p in top]}


class KLTest(unittest.TestCase):
    def test_identical_distributions_have_zero_kl(self):
        top = [("a", 0.6), ("b", 0.4)]
        self.assertAlmostEqual(qg.kl_topk([(t, math.log(p)) for t, p in top],
                                           [(t, math.log(p)) for t, p in top]), 0.0, places=9)

    def test_two_token_hand_calculation(self):
        # P = (0.9, 0.1), Q = (0.5, 0.5): 0.9 ln(1.8) + 0.1 ln(0.2) = 0.36812...
        p = [("x", math.log(0.9)), ("y", math.log(0.1))]
        q = [("x", math.log(0.5)), ("y", math.log(0.5))]
        self.assertAlmostEqual(qg.kl_topk(p, q), 0.9 * math.log(1.8) + 0.1 * math.log(0.2), places=9)
        self.assertAlmostEqual(qg.kl_topk(p, q), 0.368, places=3)

    def test_token_missing_from_candidate_list_gives_finite_kl(self):
        p = [("x", math.log(0.7)), ("y", math.log(0.3))]
        q = [("x", math.log(0.99))]
        kl = qg.kl_topk(p, q)
        self.assertTrue(math.isfinite(kl))
        self.assertGreater(kl, 0.0)

    def test_aligned_len_stops_at_first_difference(self):
        a = [{"token": t} for t in "abcd"]
        b = [{"token": t} for t in "abxd"]
        self.assertEqual(qg.aligned_len(a, b), 2)
        self.assertEqual(qg.aligned_len(a, a), 4)
        self.assertEqual(qg.aligned_len(a, a[:2]), 2)


class CompareAnswerTest(unittest.TestCase):
    def test_same_run_has_zero_kl_over_full_length(self):
        tokens = [pos("a", [("a", 0.8), ("b", 0.2)]), pos("b", [("b", 0.6), ("a", 0.4)])]
        s = qg.compare_answer({"text": "ab", "tokens": tokens}, {"text": "ab", "tokens": tokens})
        self.assertEqual(s["aligned_n"], 2)
        self.assertAlmostEqual(s["kl_mean"], 0.0, places=9)
        self.assertTrue(s["same_text"])

    def test_first_token_kl_is_available_even_when_text_diverges_at_once(self):
        a = [pos("a", [("a", 0.9), ("b", 0.1)])]
        b = [pos("b", [("a", 0.5), ("b", 0.5)])]
        s = qg.compare_answer({"text": "a", "tokens": a}, {"text": "b", "tokens": b})
        self.assertEqual(s["aligned_n"], 0)
        self.assertIsNone(s["kl_mean"])
        self.assertIsNotNone(s["first_kl"])
        self.assertAlmostEqual(s["first_kl"], 0.368, places=3)

    def test_empty_answer(self):
        s = qg.compare_answer({"text": "", "tokens": []}, {"text": "x", "tokens": [pos("x", [("x", 1.0)])]})
        self.assertIsNone(s["first_kl"])


class GateVerdictTest(unittest.TestCase):
    def test_pass_when_candidate_matches_noise_floor(self):
        ok, reasons = qg.gate_verdict(0.0, 0.0, 0.0, [0.0, 0.001], True)
        self.assertTrue(ok, reasons)

    def test_mean_limit_uses_slack_when_floor_is_zero(self):
        self.assertTrue(qg.gate_verdict(0.0, 0.0, 0.004, [0.0], True)[0])
        self.assertFalse(qg.gate_verdict(0.0, 0.0, 0.006, [0.0], True)[0])

    def test_mean_limit_scales_with_floor(self):
        self.assertTrue(qg.gate_verdict(0.1, 0.0, 0.2, [0.0], True)[0])
        self.assertFalse(qg.gate_verdict(0.1, 0.0, 0.21, [0.0], True)[0])

    def test_missing_needle_fails(self):
        ok, reasons = qg.gate_verdict(0.0, 0.0, 0.0, [0.0], False)
        self.assertFalse(ok)
        self.assertTrue(any("needle" in r for r in reasons))

    def test_first_token_bound(self):
        self.assertTrue(qg.gate_verdict(0.0, 0.0, 0.0, [0.02], True)[0])
        self.assertFalse(qg.gate_verdict(0.0, 0.0, 0.0, [0.021], True)[0])
        self.assertFalse(qg.gate_verdict(0.0, 0.01, 0.0, [0.03], True)[0])  # limit max(0.02, 0.02) = 0.02


class GateCommandTest(unittest.TestCase):
    def _write(self, d, name, answers):
        p = Path(d) / name
        p.write_text(json.dumps({"tag": name, "answers": answers}))
        return str(p)

    def test_identical_runs_pass_end_to_end(self):
        tokens = [pos("a", [("a", 0.8), ("b", 0.2)])]
        ans = {"explain_de": {"text": "a", "tokens": tokens},
               qg.NEEDLE_KEY: {"text": qg.NEEDLE, "tokens": tokens, "found": True}}
        with tempfile.TemporaryDirectory() as d:
            files = [self._write(d, f"{n}.json", ans) for n in ("a1", "a2", "b")]
            self.assertEqual(qg.gate(*files), 0)

    def test_candidate_with_missing_needle_fails_end_to_end(self):
        tokens = [pos("a", [("a", 0.8), ("b", 0.2)])]
        good = {"explain_de": {"text": "a", "tokens": tokens},
                qg.NEEDLE_KEY: {"text": qg.NEEDLE, "tokens": tokens, "found": True}}
        bad = dict(good, **{qg.NEEDLE_KEY: {"text": "no", "tokens": tokens, "found": False}})
        with tempfile.TemporaryDirectory() as d:
            a1, a2 = self._write(d, "a1.json", good), self._write(d, "a2.json", good)
            b = self._write(d, "b.json", bad)
            self.assertEqual(qg.gate(a1, a2, b), 1)


if __name__ == '__main__':
    unittest.main()
