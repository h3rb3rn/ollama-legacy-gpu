#!/usr/bin/env python3
"""
quality-gate.py — does a change alter the model's token distributions, not just its greedy text?

compare-batch-quality.py asks "is the greedy text identical?". This script also compares the
probabilities the model assigns to each token, using Ollama's logprobs (top 20 per position).

  run      <ssh-host> <port> <model> <out.json> [tag]   greedy answers with logprobs
  compare  <a.json> <b.json>                            per prompt: KL(a || b)
  gate     <a1.json> <a2.json> <b.json>                 PASS/FAIL for candidate b

Method
  * KL(A || B) per position, over the union of both top-20 lists. Probability mass outside a
    top-20 list is spread evenly over V_REST tokens; both sides are renormalised over the union.
  * Position 0 is always comparable: both runs see the same prompt there, even if the texts
    diverge later. Later positions are compared only up to the first differing token, because the
    distributions after that point are conditioned on different text.
  * Limitation: these are greedy runs, not teacher forcing. Strata forces the same token sequence
    through both models; the Ollama API does not, so only the agreeing prefix is measured.

Gate (gate_verdict). floor = the same statistics for A1 vs A2, i.e. the run-to-run noise of the
baseline. B passes when
  mean KL(A1, B) over prompts <= max(2 * floor_mean, floor_mean + 0.005)
  the needle is found in B
  first-token KL(A1, B) <= max(2 * floor_first, 0.02) for every prompt
"""
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_batch_quality", SCRIPTS / "compare-batch-quality.py")
_cbq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cbq)
PROMPTS, NEEDLE, needle_prompt = _cbq.PROMPTS, _cbq.NEEDLE, _cbq.needle_prompt

SSH_KEY = Path.home() / ".ssh" / "claude"
TOP_K = 20              # Ollama accepts 0..20
V_REST = 1000           # tokens that share the probability mass outside the top-k list
EPS = 1e-10
NEEDLE_KEY = "needle_25k"
NEEDLE_TOKENS = 60
ANSWER_TOKENS = 200
MEAN_SLACK = 0.005      # absolute slack on the mean KL, so a zero floor still allows tiny noise
FIRST_FLOOR = 0.02      # absolute bound on the first-token KL


# ── Pure math (no I/O, unit-tested) ──────────────────────────────────────────

def _top_of(entry):
    """[(token, logprob), ...] for one position; falls back to the chosen token alone."""
    top = entry.get("top") or []
    if top:
        return [(t, lp) for t, lp in top]
    return [(entry["token"], entry["logprob"])]


def _dist(top):
    probs = {t: math.exp(lp) for t, lp in top}
    rest = max(0.0, 1.0 - sum(probs.values()))
    return probs, rest / V_REST


def kl_topk(p_top, q_top):
    """KL(P || Q) in nats over the union of two top-k lists. Each list is [(token, logprob), ...]."""
    p, p_rest = _dist(p_top)
    q, q_rest = _dist(q_top)
    support = set(p) | set(q)
    P = {t: max(p.get(t, p_rest), EPS) for t in support}
    Q = {t: max(q.get(t, q_rest), EPS) for t in support}
    zp, zq = sum(P.values()), sum(Q.values())
    return sum((P[t] / zp) * math.log((P[t] / zp) / (Q[t] / zq)) for t in support)


def aligned_len(a_tokens, b_tokens):
    """Number of leading positions where both greedy runs produced the same token."""
    for i in range(min(len(a_tokens), len(b_tokens))):
        if a_tokens[i]["token"] != b_tokens[i]["token"]:
            return i
    return min(len(a_tokens), len(b_tokens))


def compare_answer(a, b):
    """Statistics for one prompt. a is the reference (P), b the candidate (Q)."""
    at, bt = a["tokens"], b["tokens"]
    res = {"same_text": a["text"] == b["text"], "aligned_n": 0, "kl_mean": None,
           "kl_p99": None, "kl_max": None, "first_kl": None}
    if not at or not bt:
        return res
    res["first_kl"] = kl_topk(_top_of(at[0]), _top_of(bt[0]))
    n = aligned_len(at, bt)
    res["aligned_n"] = n
    if n == 0:
        return res
    kls = sorted(kl_topk(_top_of(at[i]), _top_of(bt[i])) for i in range(n))
    res["kl_mean"] = sum(kls) / n
    res["kl_p99"] = kls[min(n - 1, int(0.99 * n))]
    res["kl_max"] = kls[-1]
    return res


def gate_verdict(floor_mean, floor_first, b_mean, b_firsts, needle_found):
    """Return (passed, reasons). Pure function so the thresholds can be tested."""
    reasons = []
    mean_limit = max(2 * floor_mean, floor_mean + MEAN_SLACK)
    if b_mean > mean_limit:
        reasons.append(f"mean KL {b_mean:.4f} > {mean_limit:.4f}")
    if not needle_found:
        reasons.append("needle not found")
    first_limit = max(2 * floor_first, FIRST_FLOOR)
    worst = max((f for f in b_firsts if f is not None), default=0.0)
    if worst > first_limit:
        reasons.append(f"first-token KL {worst:.4f} > {first_limit:.4f}")
    return (not reasons), reasons


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


# ── Running the model ────────────────────────────────────────────────────────

def ask(host, port, model, prompt, num_predict):
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "logprobs": True, "top_logprobs": TOP_K,
                       # temperature 0 is greedy; top_k 1 is deliberately not set, it would
                       # cut the candidate list the logprobs are taken from
                       "options": {"temperature": 0, "seed": 1, "num_predict": num_predict}})
    out = subprocess.run(["ssh", "-i", str(SSH_KEY), f"philipp@{host}",
                          f"curl -s -m 1800 localhost:{port}/api/chat -d @-"],
                         input=body, capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    if "error" in d:
        raise RuntimeError(d["error"])
    lps = d.get("logprobs")
    if not lps:
        raise RuntimeError("response has no logprobs: the server did not return them")
    tokens = [{"token": e["token"], "logprob": e["logprob"],
               "top": [[t["token"], t["logprob"]] for t in e.get("top_logprobs", [])]}
              for e in lps]
    return d["message"]["content"], d.get("prompt_eval_count"), d.get("eval_count"), tokens


def run(host, port, model, out, tag=""):
    res = {"tag": tag, "answers": {}}
    jobs = [(name, prompt, ANSWER_TOKENS) for name, prompt in PROMPTS.items()]
    jobs.append((NEEDLE_KEY, needle_prompt(), NEEDLE_TOKENS))
    for name, prompt, n_pred in jobs:
        text, n_prompt, n_eval, tokens = ask(host, port, model, prompt, n_pred)
        entry = {"text": text, "prompt_tokens": n_prompt, "eval_count": n_eval, "tokens": tokens}
        if name == NEEDLE_KEY:
            entry["found"] = NEEDLE in text
        res["answers"][name] = entry
        print(f"{name}: {len(tokens)} tokens, prompt_tokens={n_prompt}")
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False)


# ── Reports ──────────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        return json.load(f)["answers"]


def _fmt(x):
    return "-" if x is None else f"{x:.4f}"


def compare(fa, fb):
    a, b = _load(fa), _load(fb)
    print(f"{'prompt':13} {'same':5} {'aligned':>9} {'kl_mean':>9} {'kl_p99':>9} {'kl_max':>9} {'first_kl':>9}")
    for name in a:
        s = compare_answer(a[name], b[name])
        print(f"{name:13} {str(s['same_text']):5} {s['aligned_n']:>4}/{len(a[name]['tokens']):<4} "
              f"{_fmt(s['kl_mean']):>9} {_fmt(s['kl_p99']):>9} {_fmt(s['kl_max']):>9} {_fmt(s['first_kl']):>9}")


def gate(f_a1, f_a2, f_b):
    a1, a2, b = _load(f_a1), _load(f_a2), _load(f_b)
    noise = {n: compare_answer(a1[n], a2[n]) for n in a1}
    cand = {n: compare_answer(a1[n], b[n]) for n in a1}
    floor_mean = _mean(s["kl_mean"] for s in noise.values())
    floor_first = _mean(s["first_kl"] for s in noise.values())
    b_mean = _mean(s["kl_mean"] for s in cand.values())
    b_firsts = [s["first_kl"] for s in cand.values()]
    needle_found = bool(b.get(NEEDLE_KEY, {}).get("found"))
    print(f"floor (A1 vs A2): mean KL {floor_mean:.4f}, first-token KL {floor_first:.4f}")
    print(f"candidate (A1 vs B): mean KL {b_mean:.4f}, needle found: {needle_found}")
    for n, s in cand.items():
        print(f"  {n:13} kl_mean={_fmt(s['kl_mean'])} first_kl={_fmt(s['first_kl'])} aligned={s['aligned_n']}")
    passed, reasons = gate_verdict(floor_mean, floor_first, b_mean, b_firsts, needle_found)
    print("PASS" if passed else "FAIL: " + "; ".join(reasons))
    return 0 if passed else 1


def main(argv):
    if len(argv) >= 6 and argv[1] == "run":
        run(*argv[2:6], *(argv[6:7]))
        return 0
    if len(argv) == 4 and argv[1] == "compare":
        compare(argv[2], argv[3])
        return 0
    if len(argv) == 5 and argv[1] == "gate":
        return gate(argv[2], argv[3], argv[4])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
