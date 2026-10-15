#!/usr/bin/env python3
"""
bench-agent-turns.py — how much prompt is read again in agent-style conversations?

Four deterministic scenarios, each with its own system prompt (so no scenario reuses another's
cache). Requests go over SSH to /api/chat, greedy, thinking off, short answers, so the numbers
measure prompt handling, not generation.

  S1 append       8 turns; every turn appends the real answer and a new user message
  S2 edit-last    the S1 history after turn 8, then the last user message replaced 3 times
  S3 branch       the S1 history after turn 4, then 3 different next messages from that point
  S4 interleave   two conversations, turns alternating on the same server

Per request the script keeps prompt_eval_count (tokens actually read), wall time, and the
characters sent. reread_estimate = sum(prompt_eval_count) / sum(chars / CHARS_PER_TOKEN). The
token count is estimated from characters because the tokenizer is not available here; use it to
compare settings on the same machine, not as an absolute number.

  run <ssh-host> <port> <model> <out.json> [tag] [--system-tokens N] [--turn-tokens N]
"""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

SSH_KEY = Path.home() / ".ssh" / "claude"
CHARS_PER_TOKEN = 4
VOCAB = ["alpha", "beta", "gamma", "delta", "robot", "server", "kernel", "memory", "stream", "token",
         "cache", "layer", "agent", "tool", "result", "file", "plan", "check", "build", "test"]
ANSWER_TOKENS = 64


def words(rnd, tokens):
    n = max(1, int(tokens * CHARS_PER_TOKEN / 6))   # about 6 characters per word incl. space
    return " ".join(rnd.choice(VOCAB) for _ in range(n))


def chat(host, port, model, messages):
    body = json.dumps({"model": model, "stream": False, "think": False, "messages": messages,
                       "options": {"temperature": 0, "seed": 1, "num_predict": ANSWER_TOKENS}})
    t0 = time.monotonic()
    out = subprocess.run(["ssh", "-i", str(SSH_KEY), f"philipp@{host}",
                          f"curl -s -m 1800 localhost:{port}/api/chat -d @-"],
                         input=body, capture_output=True, text=True, check=True).stdout
    wall = time.monotonic() - t0
    d = json.loads(out)
    if "error" in d:
        raise RuntimeError(d["error"])
    chars = sum(len(m["content"]) for m in messages)
    rec = {"prompt_eval_count": d.get("prompt_eval_count", 0), "eval_count": d.get("eval_count", 0),
           "wall_s": round(wall, 3), "chars": chars}
    return d["message"]["content"], rec


class Conversation:
    """One conversation: a system prompt plus alternating user and assistant messages."""

    def __init__(self, seed, system_tokens, turn_tokens):
        self.rnd = random.Random(seed)
        self.turn_tokens = turn_tokens
        self.messages = [{"role": "system", "content": words(self.rnd, system_tokens)}]

    def user_text(self):
        return words(self.rnd, self.turn_tokens)

    def snapshot(self):
        return list(self.messages)


def turn(host, port, model, conv, user_msg, log, label, keep=True):
    msgs = conv.snapshot() + [{"role": "user", "content": user_msg}]
    answer, rec = chat(host, port, model, msgs)
    rec["label"] = label
    log.append(rec)
    if keep:
        conv.messages = msgs + [{"role": "assistant", "content": answer}]
    return answer


def scenario_append(host, port, model, s, log, sys_tok, turn_tok):
    conv = Conversation(seed=101 + s, system_tokens=sys_tok, turn_tokens=turn_tok)
    for i in range(1, 9):
        turn(host, port, model, conv, conv.user_text(), log, f"append-{i}")


def scenario_edit_last(host, port, model, s, log, sys_tok, turn_tok):
    conv = Conversation(seed=201 + s, system_tokens=sys_tok, turn_tokens=turn_tok)
    for i in range(1, 9):
        turn(host, port, model, conv, conv.user_text(), log, f"edit-prefix-{i}")
    base = conv.snapshot()
    for k in range(3):
        conv.messages = base
        turn(host, port, model, conv, conv.user_text(), log, f"edit-last-{k + 1}", keep=False)


def scenario_branch(host, port, model, s, log, sys_tok, turn_tok):
    conv = Conversation(seed=301 + s, system_tokens=sys_tok, turn_tokens=turn_tok)
    for i in range(1, 5):
        turn(host, port, model, conv, conv.user_text(), log, f"branch-prefix-{i}")
    base = conv.snapshot()
    for k in range(3):
        conv.messages = base
        turn(host, port, model, conv, conv.user_text(), log, f"branch-{k + 1}", keep=False)


def scenario_interleave(host, port, model, s, log, sys_tok, turn_tok):
    a = Conversation(seed=401 + s, system_tokens=sys_tok, turn_tokens=turn_tok)
    b = Conversation(seed=501 + s, system_tokens=sys_tok, turn_tokens=turn_tok)
    for i in range(1, 5):
        turn(host, port, model, a, a.user_text(), log, f"interleave-a-{i}")
        turn(host, port, model, b, b.user_text(), log, f"interleave-b-{i}")


SCENARIOS = [("S1-append", scenario_append), ("S2-edit-last", scenario_edit_last),
             ("S3-branch", scenario_branch), ("S4-interleave", scenario_interleave)]


def summarise(log):
    total_read = sum(r["prompt_eval_count"] for r in log)
    total_est = sum(r["chars"] for r in log) / CHARS_PER_TOKEN
    return {"requests": len(log), "prompt_read_tokens": total_read,
            "prompt_est_tokens": round(total_est), "reread_estimate": round(total_read / total_est, 4)
            if total_est else None, "wall_s": round(sum(r["wall_s"] for r in log), 2)}


def run(host, port, model, out, tag="", system_tokens=1000, turn_tokens=300):
    result = {"tag": tag, "model": model, "system_tokens": system_tokens, "turn_tokens": turn_tokens,
              "scenarios": {}}
    for s, (name, fn) in enumerate(SCENARIOS):
        log = []
        fn(host, port, model, s, log, system_tokens, turn_tokens)
        result["scenarios"][name] = {"summary": summarise(log), "requests": log}
        print(f"{name}: {summarise(log)}")
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {"--system-tokens": 1000, "--turn-tokens": 300}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            opts[flag] = int(args[i + 1])
            del args[i:i + 2]
    if len(args) >= 4 and args[0] == "run":
        run(*args[1:5], *(args[5:6]), system_tokens=opts["--system-tokens"], turn_tokens=opts["--turn-tokens"])
    else:
        print(__doc__)
        sys.exit(2)
