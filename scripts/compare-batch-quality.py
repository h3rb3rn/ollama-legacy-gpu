#!/usr/bin/env python3
"""
compare-batch-quality.py — does the batch size change what the model says?

Batch size only changes how llama.cpp tiles the prompt (the reduction order of the matmuls), not
the model, so outputs should agree up to floating-point noise. This script checks that instead of
assuming it:

  run      <ssh-host> <port> <model> <out.json> [tag]   greedy answers (temperature 0, fixed seed) to a
                                                        fixed prompt set incl. a ~18k-token needle test
                                                        (key `needle_25k` is the historical name)
  compare  <a.json> <b.json>                            per prompt: identical?, first differing character,
                                                        needle found?

Do three runs on freshly loaded containers (prompt-cache reuse would make a repeat trivially identical):
  A1 = batch 512, B = batch 64, A2 = batch 512 again.  A1-vs-A2 is the run-to-run noise floor; B-vs-A1
  only counts as a batch effect if it is worse than A1-vs-A2.
"""
import json
import random
import subprocess
import sys

SSH_KEY = "~/.ssh/claude"

PROMPTS = {
    "explain_de": "Erkläre in genau fünf Sätzen, wie ein Transformer-Sprachmodell Text verarbeitet.",
    "code_py": "Write a Python function merge_intervals(intervals) that merges overlapping intervals. "
               "Include a docstring and two doctests. Output only code.",
    "math_de": "Was ist 17*23 + 144/12? Rechne Schritt für Schritt und nenne am Ende das Ergebnis.",
    "translate": "Translate to English: 'Der Server startet das Modell, verteilt die Schichten auf die "
                 "Grafikkarten und beantwortet danach jede Anfrage in wenigen Sekunden.'",
    "json_extract": "Extract name, city and year as JSON from: 'Anna Berger zog 2019 von Kassel nach Leipzig.' "
                    "Output only JSON.",
}
NEEDLE = "BLAUER-FALKE-4711"


def needle_prompt(words=18000, seed=7):
    rnd = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "robot", "server", "kernel", "memory", "stream", "token",
             "cache", "layer"]
    body = [rnd.choice(vocab) for _ in range(words)]
    body.insert(len(body) // 2, f"Das geheime Passwort lautet {NEEDLE}.")
    return " ".join(body) + "\n\nWie lautet das geheime Passwort? Antworte nur mit dem Passwort."


def ask(host, port, model, prompt, num_predict):
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": 0, "seed": 1, "top_k": 1, "num_predict": num_predict}})
    out = subprocess.run(["ssh", "-i", SSH_KEY.replace("~", __import__("os").path.expanduser("~")),
                          f"philipp@{host}", f"curl -s -m 1800 localhost:{port}/api/chat -d @-"],
                         input=body, capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    if "error" in d:
        raise RuntimeError(d["error"])
    return d["message"]["content"], d.get("prompt_eval_count")


def run(host, port, model, out, tag=""):
    res = {"tag": tag, "answers": {}}
    for name, prompt in PROMPTS.items():
        text, n = ask(host, port, model, prompt, 200)
        res["answers"][name] = {"text": text, "prompt_tokens": n}
        print(f"{name}: {len(text)} chars")
    text, n = ask(host, port, model, needle_prompt(), 60)
    res["answers"]["needle_25k"] = {"text": text, "prompt_tokens": n, "found": NEEDLE in text}
    print(f"needle_25k: prompt_tokens={n} found={NEEDLE in text}")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(fa, fb):
    a, b = json.load(open(fa))["answers"], json.load(open(fb))["answers"]
    same = 0
    for name in a:
        d = first_diff(a[name]["text"], b[name]["text"])
        same += d is None
        extra = ""
        if "found" in a[name]:
            extra = f" needle: {a[name]['found']} vs {b[name]['found']}"
        print(f"{name:13} {'identisch' if d is None else f'erste Abweichung bei Zeichen {d} von {len(a[name]['text'])}'}{extra}")
    print(f"identisch: {same}/{len(a)}")


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "run":
        run(*sys.argv[2:6], *(sys.argv[6:7]))
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
