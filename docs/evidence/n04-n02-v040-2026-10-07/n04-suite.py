#!/usr/bin/env python3
"""v0.40.0 candidate on N04-RTX: replicas of the 11436 (4x M10) and 11442 (2x M60) configuration. Production containers untouched."""
import json, re, subprocess, time, urllib.request, random, os
OUT = os.environ.get("OUTDIR", "/home/philipp/n04-suite"); os.makedirs(OUT, exist_ok=True)
RES = open(f"{OUT}/results.jsonl", "a", buffering=1); LOG = open(f"{OUT}/progress.log", "a", buffering=1)
IMG = os.environ.get("IMG", "ghcr.io/h3rb3rn/ollama-legacy:candidate-37600164484-1-cuda12-maxwell-native")
M10 = "GPU-f668b685-3c6d-0fc7-1f85-960a0607c561,GPU-086bd19b-5a4c-a6dc-8c82-9c1a5276b0b2,GPU-7b3bd39b-b948-09a6-8ef9-756406efdb99,GPU-113fd30d-cb6f-ad02-0803-2c2b9053709d"
M60 = "GPU-d34095b6-ff1d-6abb-9605-94a1b120f7b3,GPU-2dc77b8a-fb51-6cfa-fd0c-2bb951f55b98"
N10 = "ollama-test-" + os.environ.get("TAG", "v040") + "-m10"
N60 = "ollama-test-" + os.environ.get("TAG", "v040") + "-m60"
def log(m): LOG.write(time.strftime("%H:%M:%S ") + m + "\n")
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def rec(**k):
    k["ts"] = time.strftime("%FT%T"); RES.write(json.dumps(k, ensure_ascii=False) + "\n"); log("RESULT " + json.dumps(k, ensure_ascii=False)[:260])
def post(port, path, body, to=1800):
    r = urllib.request.Request(f"http://localhost:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
def start(name, port, uuids, env):
    sh(f"docker rm -f {name} >/dev/null 2>&1")
    e = " ".join(f"-e {k}={v}" for k, v in env.items())
    sh(f"docker run -d --name {name} -p 127.0.0.1:{port}:11434 -v /home/philipp/v040-store:/root/.ollama/models -v /opt/ollama/models/models/blobs:/root/.ollama/models/blobs:ro "
       f"-e CUDA_VISIBLE_DEVICES={uuids} {e} --gpus '\"device={uuids}\"' {IMG} serve")
    for _ in range(120):
        try: urllib.request.urlopen(f"http://localhost:{port}/api/version", timeout=3); return
        except Exception: time.sleep(2)
    raise RuntimeError("api not up")
def info(name):
    lg = sh(f"docker logs {name} 2>&1"); g = lambda p: (re.findall(p, lg) or [""])[-1]
    return {"layers": g(r"offloaded (\d+/\d+) layers"), "n_ctx": g(r"llama_context: n_ctx\s+= (\d+)"), "n_batch": g(r"llama_context: n_batch\s+= (\d+)"),
            "cuda_errors": len(re.findall(r"CUDA error", lg)), "spec": "--spec" in sh(f"docker exec {name} sh -c 'ps aux | grep llama-server | grep -v grep'"),
            "version": g(r"Listening on [^\n]*\(version ([^)]+)\)")}
def gpu_used(): return sh("nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits")
def test(name, port, model, tag, uuids, long_prompt=False):
    t0 = time.time()
    try:
        post(port, "/api/generate", {"model": model, "keep_alive": "10m"}, 1800); load_s = round(time.time() - t0, 1)
        used = [int(l.split(",")[2]) for l in gpu_used().split("\n") if l.split(",")[1].strip() in uuids.split(",")]
        dec = []
        for _ in range(2):
            d = post(port, "/api/generate", {"model": model, "prompt": "Write a long story about a robot.", "stream": False, "options": {"num_predict": 120, "temperature": 0, "seed": 1}}, 900)
            dec.append(round(d["eval_count"] / d["eval_duration"] * 1e9, 1))
        rng = random.Random(hash(tag) & 0xffff); w = "Haus Garten Lampe Zug Fenster Wasser Berg river engine paper window market garden".split()
        d = post(port, "/api/generate", {"model": model, "prompt": " ".join(rng.choice(w) for _ in range(2500)) + "\nSummarize in one sentence.", "stream": False, "options": {"num_predict": 8, "temperature": 0}}, 1800)
        pre = round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 1)
        c = post(port, "/api/chat", {"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}], "options": {"num_predict": 400, "temperature": 0}}, 900)
        ans = (c["message"]["content"] or "")[-120:].replace("\n", " ")
        rec(test=tag, model=model, ok=True, load_s=load_s, decode=dec, prefill=pre, vram_used_mib=used, answer_tail=ans, **info(name))
    except Exception as e:
        rec(test=tag, model=model, ok=False, error=str(e)[:200], **info(name))
def run():
    # replica of ollama-tesla-bonsai (:11436): 4x Tesla M10, FA on, KV q4_0, ctx 190000, spread, autodetect off
    env10 = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_CONTEXT_LENGTH": "190000", "OLLAMA_NUM_PARALLEL": "1",
             "OLLAMA_SCHED_SPREAD": "true", "OLLAMA_LOAD_TIMEOUT": "30m", "OLLAMA_GPU_AUTODETECT": "0", "OLLAMA_AUTO_OPTIMIZE": "0", "OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_INTERNAL_PORT": "11434"}
    log("replica 11436 (4x M10)"); start(N10, 21436, M10, env10)
    test(N10, 21436, "bonsai2:27b-pq2_0", "m10-bonsai-expected-fail", M10)
    sh(f"docker restart {N10} >/dev/null"); time.sleep(20)
    test(N10, 21436, "qwen3.5:9b", "m10-qwen3.5-9b", M10)
    test(N10, 21436, "qwen3.6:35b", "m10-qwen3.6-35b", M10)
    test(N10, 21436, "hf.co/h3rb3rn/sovereign-judge-olmo31-32b:Q4_K_M", "m10-judge-olmo-32b", M10)
    sh(f"docker rm -f {N10} >/dev/null 2>&1")
    # replica of ollama-m60-guard (:11442): 2x Tesla M60, ctx 32768, FA on, KV q4_0, no mmproj offload, no spread
    env60 = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "1",
             "OLLAMA_SCHED_SPREAD": "false", "OLLAMA_LOAD_TIMEOUT": "20m", "OLLAMA_GPU_AUTODETECT": "0", "OLLAMA_AUTO_OPTIMIZE": "0",
             "LLAMA_ARG_MMPROJ_OFFLOAD": "false", "LLAMA_ARG_THREADS": "4", "LLAMA_ARG_THREADS_BATCH": "4", "OLLAMA_GPU_OVERHEAD": "268435456", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_INTERNAL_PORT": "11434"}
    log("replica 11442 (2x M60)"); start(N60, 21442, M60, env60)
    test(N60, 21442, "llama-guard3:8b", "m60-llama-guard3-8b", M60)
    test(N60, 21442, "qwen3.5:9b", "m60-qwen3.5-9b", M60)
    test(N60, 21442, "hf.co/h3rb3rn/moe-sovereign-planner-9b:Q4_K_M", "m60-planner-9b", M60)
    sh(f"docker rm -f {N60} >/dev/null 2>&1")
if __name__ == "__main__":
    try: run()
    except Exception as e: log(f"ABORT {e!r}")
    finally:
        sh(f"docker rm -f {N10} {N60} >/dev/null 2>&1"); open(f"{OUT}/DONE", "w").write(time.strftime("%F %T")); log("DONE")
