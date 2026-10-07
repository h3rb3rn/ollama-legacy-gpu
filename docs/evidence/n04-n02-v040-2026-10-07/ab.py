#!/usr/bin/env python3
"""A/B on N02-M60 GPU0-3: deployed image (0.35.1 fork) vs v0.40.0 candidate with the deployed, tuned configuration."""
import json, os, re, subprocess, time, random, urllib.request
OUT = "/home/philipp/ab-n02"; os.makedirs(OUT, exist_ok=True)
RES = open(f"{OUT}/results.jsonl", "a", buffering=1); LOG = open(f"{OUT}/progress.log", "a", buffering=1)
A = "ollama-gaps:kolibri-20261006"
B = os.environ.get("IMG_B", "ghcr.io/h3rb3rn/ollama-legacy:candidate-37600164484-1-cuda12-maxwell-native")
ENVFILE = "/opt/deployment/ollama/llm-studio/worker-m60/.env.m60-single12"
MODEL = "qwen3.6:35b"
def log(m): LOG.write(time.strftime("%H:%M:%S ") + m + "\n")
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
uuids = [l.split("UUID: ")[1].rstrip(")") for l in sh("nvidia-smi -L").split("\n")][:4]
ids = ",".join(uuids)
env = [l for l in open(ENVFILE).read().split("\n") if l and not l.startswith("#") and not l.startswith(("CUDA_VISIBLE_DEVICES", "OLLAMA_FORCE_GPU_LAYERS", "OLLAMA_LAYER_OVERHEAD_SCALE"))]
open(f"{OUT}/ab.env", "w").write("\n".join(env) + "\n")
def post(path, body, to=1800):
    r = urllib.request.Request(f"http://localhost:21434{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
def run(label, img, rnd):
    sh("docker rm -f ab-test >/dev/null 2>&1")
    sh(f"docker run -d --name ab-test -p 127.0.0.1:21434:11434 -v /home/philipp/v040-store-n02:/root/.ollama/models -v /opt/ollama/models/models/blobs:/root/.ollama/models/blobs:ro "
       f"--env-file {OUT}/ab.env -e CUDA_VISIBLE_DEVICES={ids} --gpus '\"device={ids}\"' {img} serve")
    for _ in range(120):
        try: urllib.request.urlopen("http://localhost:21434/api/version", timeout=3); break
        except Exception: time.sleep(2)
    t0 = time.time()
    try:
        post("/api/generate", {"model": MODEL, "keep_alive": "10m"}); load_s = round(time.time() - t0, 1)
        dec = []
        for _ in range(3):
            d = post("/api/generate", {"model": MODEL, "prompt": "Write a long story about a robot.", "stream": False, "options": {"num_predict": 120, "temperature": 0, "seed": 1}})
            dec.append(round(d["eval_count"] / d["eval_duration"] * 1e9, 1))
        rng = random.Random(rnd * 7 + len(label)); w = "Haus Garten Lampe Zug Fenster Wasser Berg river engine paper window market garden".split()
        d = post("/api/generate", {"model": MODEL, "prompt": " ".join(rng.choice(w) for _ in range(2500)) + "\nSummarize in one sentence.", "stream": False, "options": {"num_predict": 8, "temperature": 0}})
        pre = round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 1)
        lg = sh("docker logs ab-test 2>&1"); g = lambda p: (re.findall(p, lg) or [""])[-1]
        cmd = sh("docker exec ab-test sh -c 'ps aux | grep llama-server | grep -v grep'")
        rec = dict(label=label, round=rnd, ok=True, load_s=load_s, decode=dec, prefill=pre, layers=g(r"offloaded (\d+/\d+) layers"), n_ctx=g(r"llama_context: n_ctx\s+= (\d+)"),
                   n_batch=g(r"llama_context: n_batch\s+= (\d+)"), spec="--spec" in cmd, host_buf=g(r"CUDA_Host compute buffer size\s+=\s+([\d.]+)"), cuda_errors=len(re.findall("CUDA error", lg)),
                   fit_s=g(r"fit.*?(\d+\.\d+) ?s"))
    except Exception as e:
        rec = dict(label=label, round=rnd, ok=False, error=str(e)[:200])
    rec["ts"] = time.strftime("%FT%T"); RES.write(json.dumps(rec) + "\n"); log("RESULT " + json.dumps(rec)[:300])
    sh("docker rm -f ab-test >/dev/null 2>&1")
sh("rm -rf /home/philipp/v040-store-n02; mkdir -p /home/philipp/v040-store-n02; cp -a /opt/ollama/models/models/manifests /home/philipp/v040-store-n02/manifests")
try:
    for rnd in (1, 2):
        run("A-deployed-0.35.1", A, rnd); run("B-v0.40.0", B, rnd)
finally:
    sh("docker rm -f ab-test >/dev/null 2>&1"); open(f"{OUT}/DONE", "w").write(time.strftime("%F %T")); log("DONE")
