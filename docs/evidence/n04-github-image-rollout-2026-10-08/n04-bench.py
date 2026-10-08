import json, re, subprocess, sys, time, random, urllib.request, os
OUT = os.environ.get("OUTFILE", "/home/philipp/n04-rollout.jsonl")
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def post(port, path, body, to=1800):
    r = urllib.request.Request(f"http://localhost:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}); return json.load(urllib.request.urlopen(r, timeout=to))
def get(port, path): return json.load(urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=60))
for spec in sys.argv[1:]:
    port, cont, model, tag = spec.split("|")
    t0 = time.time(); rec = dict(tag=tag, port=int(port), container=cont, model=model)
    try:
        post(port, "/api/generate", {"model": model, "keep_alive": "24h"}); rec["load_s"] = round(time.time() - t0, 1)
        dec = []
        for _ in range(3):
            d = post(port, "/api/generate", {"model": model, "prompt": "Write a long story about a robot.", "stream": False, "options": {"num_predict": 120, "temperature": 0, "seed": 1}}); dec.append(round(d["eval_count"] / d["eval_duration"] * 1e9, 1))
        rng = random.Random(len(tag)); w = "Haus Garten Lampe Zug Fenster Wasser Berg river engine paper window market garden".split()
        d = post(port, "/api/generate", {"model": model, "prompt": " ".join(rng.choice(w) for _ in range(2500)) + "\nSummarize in one sentence.", "stream": False, "options": {"num_predict": 8, "temperature": 0}}); pre = round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 1)
        c = post(port, "/api/chat", {"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}], "options": {"num_predict": 400, "temperature": 0}})
        lg = sh(f"docker logs {cont} 2>&1"); g = lambda p: (re.findall(p, lg) or [""])[-1]
        ps = get(port, "/api/ps")["models"][0]
        rec.update(ok=True, decode=dec, prefill=pre, layers=g(r"offloaded (\d+/\d+) layers"), n_ctx=g(r"llama_context: n_ctx\s+= (\d+)"), n_batch=g(r"llama_context: n_batch\s+= (\d+)"),
                   host_buf=g(r"CUDA_Host compute buffer size\s+=\s+([\d.]+)"), cuda_errors=len(re.findall("CUDA error", lg)), gpu_pct=round(100 * ps["size_vram"] / ps["size"], 1),
                   version=get(port, "/api/version")["version"], answer=(c["message"]["content"] or "")[-40:].replace("\n", " "))
    except Exception as e: rec.update(ok=False, error=str(e)[:200], cuda_errors=len(re.findall("CUDA error", sh(f"docker logs {cont} 2>&1"))))
    rec["ts"] = time.strftime("%FT%T"); open(OUT, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec))
