import importlib.util
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr = load('prefix_router')


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def chat(*contents, model='qwen3.6:35b'):
    roles = ['user', 'assistant']
    msgs = [{'role': roles[i % 2], 'content': c} for i, c in enumerate(contents)]
    return {'model': model, 'messages': msgs}


class PrefixKeysTest(unittest.TestCase):
    def test_same_history_gives_same_keys(self):
        a, _ = pr.prefix_keys(chat('hello there', 'reply'))
        b, _ = pr.prefix_keys(chat('hello there', 'reply'))
        self.assertEqual(a, b)

    def test_changed_first_message_changes_every_key(self):
        a, _ = pr.prefix_keys(chat('hello there', 'reply'))
        b, _ = pr.prefix_keys(chat('HELLO there', 'reply'))
        self.assertNotEqual(a[0][0], b[0][0])
        self.assertNotEqual(a[1][0], b[1][0])

    def test_extended_conversation_shares_leading_keys(self):
        a, _ = pr.prefix_keys(chat('one', 'two'))
        b, _ = pr.prefix_keys(chat('one', 'two', 'three'))
        self.assertEqual(a, b[:2])

    def test_chars_and_total(self):
        keys, total = pr.prefix_keys(chat('abc', 'de'))
        self.assertEqual([upto for _, upto in keys], [3, 5])
        self.assertEqual(total, 5)

    def test_tools_are_part_of_the_prefix(self):
        a, _ = pr.prefix_keys(dict(chat('x'), tools=[{'name': 'a'}]))
        b, _ = pr.prefix_keys(dict(chat('x'), tools=[{'name': 'b'}]))
        self.assertNotEqual(a[-1][0], b[-1][0])

    def test_plain_prompt_uses_full_blocks(self):
        text = 'x' * (pr.BLOCK_CHARS * 2 + 10)
        keys, total = pr.prefix_keys({'prompt': text})
        self.assertEqual(len(keys), 2)
        self.assertEqual(total, len(text))

    def test_non_dict_or_empty(self):
        self.assertEqual(pr.prefix_keys(None), ([], 0))
        self.assertEqual(pr.prefix_keys({'messages': []}), ([], 0))


class ChooseTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.r = pr.PrefixRouter(['http://a', 'http://b'], clock=self.clock)

    def _serve(self, backend, body):
        keys, total = pr.prefix_keys(body)
        self.r.begin(backend)
        self.r.end(backend, keys, ok=True)
        return keys, total

    def test_same_conversation_goes_back_to_same_backend(self):
        turn1 = chat('a' * 4000, 'reply')
        b = self.r.choose('qwen3.6:35b', *pr.prefix_keys(turn1))
        self._serve(b, turn1)
        turn2 = chat('a' * 4000, 'reply', 'next question')
        self.assertEqual(self.r.choose('qwen3.6:35b', *pr.prefix_keys(turn2)), b)

    def test_new_conversation_goes_to_less_loaded_backend(self):
        self.r.begin('http://a')
        self.assertEqual(self.r.choose('qwen3.6:35b', *pr.prefix_keys(chat('new'))), 'http://b')

    def test_heavy_load_beats_affinity_for_short_prefix(self):
        turn1 = chat('short prompt', 'reply')
        self._serve('http://a', turn1)
        for _ in range(2):
            self.r.begin('http://a')   # two requests now running on a
        turn2 = chat('short prompt', 'reply', 'more')
        self.assertEqual(self.r.choose('qwen3.6:35b', *pr.prefix_keys(turn2)), 'http://b')

    def test_backend_without_the_model_is_never_chosen(self):
        self.r.set_models('http://a', ['other:latest'])
        self.r.set_models('http://b', ['qwen3.6:35b'])
        self.r.begin('http://b')
        self.assertEqual(self.r.choose('qwen3.6:35b', [], 10), 'http://b')

    def test_model_name_without_tag_matches_latest(self):
        self.r.set_models('http://a', ['qwen3.6:latest'])
        self.r.set_models('http://b', ['other:latest'])
        self.assertEqual(self.r.choose('qwen3.6', [], 10), 'http://a')

    def test_down_backend_is_skipped_then_used_again(self):
        self.r.mark_down('http://a')
        self.assertEqual(self.r.choose('m', [], 10), 'http://b')
        self.clock.t += pr.DOWN_S + 1
        self.assertEqual(self.r.choose('m', [], 10), 'http://a')

    def test_exclude_and_none_left(self):
        self.assertEqual(self.r.choose('m', [], 10, exclude={'http://a'}), 'http://b')
        self.assertIsNone(self.r.choose('m', [], 10, exclude={'http://a', 'http://b'}))

    def test_failed_request_does_not_index_prefixes(self):
        body = chat('x' * 500, 'reply')
        keys, total = pr.prefix_keys(body)
        self.r.begin('http://a')
        self.r.end('http://a', keys, ok=False)
        self.assertEqual(self.r.snapshot()['http://a']['prefixes'], 0)

    def test_lru_is_bounded(self):
        old = pr.INDEX_MAX
        try:
            pr.INDEX_MAX = 5
            for i in range(20):
                keys, _ = pr.prefix_keys(chat(f'conversation {i}'))
                self.r.begin('http://a')
                self.r.end('http://a', keys, ok=True)
            self.assertEqual(self.r.snapshot()['http://a']['prefixes'], 5)
        finally:
            pr.INDEX_MAX = old

    def test_needs_a_backend(self):
        with self.assertRaises(ValueError):
            pr.PrefixRouter([])


if __name__ == '__main__':
    unittest.main()
