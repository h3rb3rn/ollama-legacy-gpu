import importlib.util
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


turns = load('bench-agent-turns')
conc = load('bench-concurrency')


class AgentTurnsHelpers(unittest.TestCase):
    def test_summary_reread_estimate(self):
        log = [{"prompt_eval_count": 100, "chars": 400, "wall_s": 1.0, "eval_count": 5},
               {"prompt_eval_count": 0, "chars": 400, "wall_s": 0.5, "eval_count": 5}]
        s = turns.summarise(log)
        self.assertEqual(s["requests"], 2)
        self.assertEqual(s["prompt_read_tokens"], 100)
        self.assertEqual(s["prompt_est_tokens"], 200)      # 800 chars / 4
        self.assertAlmostEqual(s["reread_estimate"], 0.5)

    def test_conversation_is_deterministic(self):
        a = turns.Conversation(seed=1, system_tokens=50, turn_tokens=20)
        b = turns.Conversation(seed=1, system_tokens=50, turn_tokens=20)
        self.assertEqual(a.messages, b.messages)
        self.assertEqual(a.user_text(), b.user_text())

    def test_system_prompt_size_follows_tokens(self):
        conv = turns.Conversation(seed=2, system_tokens=400, turn_tokens=20)
        self.assertGreater(len(conv.messages[0]["content"]), 400)


class ConcurrencyHelpers(unittest.TestCase):
    def test_percentiles(self):
        vals = [1, 2, 3, 4, 5]
        self.assertEqual(conc.percentile(vals, 0.5), 3)
        self.assertEqual(conc.percentile(vals, 0.95), 5)
        self.assertIsNone(conc.percentile([], 0.5))

    def test_conversation_words_deterministic(self):
        self.assertEqual(conc.conversation_words(3, 10), conc.conversation_words(3, 10))


if __name__ == '__main__':
    unittest.main()
