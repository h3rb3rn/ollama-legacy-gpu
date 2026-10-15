#!/usr/bin/env python3
"""
bench-concurrency.py — total throughput of one wide instance vs. several replicas.

N client threads, each with its own conversation, send requests over SSH to /api/chat. Client i
uses port ports[i % len(ports)], so one port means one instance and two ports mean two replicas
(round-robin, no prefix routing; that is what prefix_router.py adds on top).

Per request: eval_count, wall time and time to first token (streaming). The aggregate is
sum(eval_count) / wall time of the whole run, measured from the first request to the last answer.

  run <ssh-host> <ports,comma-separated> <model> <clients> <requests-per-client> <out.json> [tag]
"""
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

SSH_KEY = Path.home() / ".ssh" / "claude"
VOCAB = ["alpha", "beta", "gamma", "delta", "robot", "server", "kernel", "memory", "stream", "token"]
PROMPT_WORDS = 1200           # about 2k tokens of system prompt per client
NUM_PREDICT = 200


def conversation_words(seed, n):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(VOCAB) for _ in range(n))


def request(host, port, model, messages):
    """Stream one chat request; return (text, eval_count, ttft_s, wall_s)."""
    body = json.dumps({"model": model, "stream": True, "think": False, "messages": messages,
                       "options": {"temperature": 0, "seed": 1, "num_predict": NUM_PREDICT}})
    t0 = time.monotonic()
    proc = subprocess.Popen(["ssh", "-i", str(SSH_KEY), f"philipp@{host}",
                             f"curl -s -N -m 1800 localhost:{port}/api/chat -d @-"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdin.write(body)
    proc.stdin.close()
    text, eval_count, ttft = [], 0, None
    for line in proc.stdout:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if "error" in chunk:
            proc.wait()
            raise RuntimeError(chunk["error"])
        if chunk.get("message", {}).get("content") and ttft is None:
            ttft = time.monotonic() - t0
        text.append(chunk.get("message", {}).get("content", ""))
        if chunk.get("done"):
            eval_count = chunk.get("eval_count", 0)
    proc.wait()
    return "".join(text), eval_count, ttft, time.monotonic() - t0


def client(idx, host, port, model, n_requests, results, errors):
    messages = [{"role": "system", "content": conversation_words(7000 + idx, PROMPT_WORDS)}]
    for r in range(n_requests):
        messages.append({"role": "user", "content": f"Question {r} of client {idx}: summarise the last part."})
        try:
            text, n_eval, ttft, wall = request(host, port, model, messages)
        except Exception as e:  # keep the other clients running; the error is reported
            errors.append(f"client {idx} request {r}: {e}")
            return
        messages.append({"role": "assistant", "content": text})
        results.append({"client": idx, "port": port, "eval_count": n_eval,
                        "ttft_s": round(ttft or 0.0, 3), "wall_s": round(wall, 3)})


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def run(host, ports, model, clients, n_requests, out, tag=""):
    ports = [p.strip() for p in ports.split(",") if p.strip()]
    results, errors, threads = [], [], []
    t0 = time.monotonic()
    for i in range(clients):
        th = threading.Thread(target=client, args=(i, host, ports[i % len(ports)], model, n_requests,
                                                   results, errors))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    elapsed = time.monotonic() - t0
    tokens = sum(r["eval_count"] for r in results)
    summary = {"clients": clients, "ports": ports, "requests": len(results), "errors": errors,
               "aggregate_tok_s": round(tokens / elapsed, 2) if elapsed else None,
               "elapsed_s": round(elapsed, 2),
               "ttft_p50_s": percentile([r["ttft_s"] for r in results], 0.5),
               "wall_p50_s": percentile([r["wall_s"] for r in results], 0.5),
               "wall_p95_s": percentile([r["wall_s"] for r in results], 0.95)}
    print(json.dumps(summary, indent=1))
    Path(out).write_text(json.dumps({"tag": tag, "model": model, "summary": summary,
                                     "requests": results}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) >= 8 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7],
            *(sys.argv[8:9]))
    else:
        print(__doc__)
        sys.exit(2)
