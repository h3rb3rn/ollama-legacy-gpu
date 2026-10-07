#!/usr/bin/env python3
"""Test suite on N02-M60: GPU-count matrix, context ladder, soak. Results: results.jsonl (one JSON per measurement)."""
import json, os, random, re, subprocess, sys, threading, time, urllib.request

OUT = "/home/philipp/kol-suite"
ENVBASE = "/opt/deployment/ollama/llm-studio/worker-m60/.env.m60-single12"
COMPOSE = "/opt/deployment/ollama/llm-studio/worker-m60"
os.makedirs(OUT, exist_ok=True)
RES = open(f"{OUT}/results.jsonl", "a", buffering=1)
LOG = open(f"{OUT}/progress.log", "a", buffering=1)
KOL = "hf.co/Hob-forge/Kolibri-1-GGUF:Q4_K_M"
QWEN = "qwen3.6:35b"
URL = "http://localhost:11434"

def log(m):
    LOG.write(time.strftime("%H:%M:%S ") + m + "\n")

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()

def record(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RES.write(json.dumps(kw, ensure_ascii=False) + "\n")
    log("RESULT " + json.dumps(kw, ensure_ascii=False)[:300])

UUIDS = [l.split(",")[1].strip() for l in open("/tmp/uuids.csv").read().split("\n") if l.strip()]
assert len(UUIDS) == 12

def post(path, body, timeout=14400):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def gpu_mem():
    out = sh("nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader,nounits")
    return [tuple(int(x) for x in l.split(",")) for l in out.split("\n") if l]

def stop_all():
    sh("docker rm -f ollama-m60-test >/dev/null 2>&1; docker stop ollama-m60-12 >/dev/null 2>&1")

def start(image, n, force=None, tag=""):
    sh("docker rm -f ollama-m60-test >/dev/null 2>&1")
    envs = [l for l in open(ENVBASE).read().split("\n") if l and not l.startswith("#")
            and not l.startswith(("CUDA_VISIBLE_DEVICES", "OLLAMA_FORCE_GPU_LAYERS", "OLLAMA_LAYER_OVERHEAD_SCALE"))]
    if force:
        envs += ["OLLAMA_FORCE_GPU_LAYERS=1", f"OLLAMA_LAYER_OVERHEAD_SCALE={force}"]
    open(f"{OUT}/test.env", "w").write("\n".join(envs) + "\n")
    ids = ",".join(UUIDS[:n])
    cmd = (f"docker run -d --name ollama-m60-test -p 11434:11434 -v /opt/ollama/models:/root/.ollama "
           f"--env-file {OUT}/test.env -e CUDA_VISIBLE_DEVICES={ids} --gpus '\"device={ids}\"' {image} serve")
    sh(cmd, check=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/api/version", timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("API not up")

def runner_info():
    lg = sh("docker logs ollama-m60-test 2>&1")
    g = lambda pat: (re.findall(pat, lg) or [""])[-1]
    return {
        "layers": g(r"offloaded (\d+/\d+) layers"),
        "n_ctx": g(r"llama_context: n_ctx\s+= (\d+)"),
        "n_batch": g(r"llama_context: n_batch\s+= (\d+)"),
        "greedy": g(r"greedy fill: ([^\n]*)")[:80],
        "cuda_errors": len(re.findall(r"CUDA error", lg)),
    }

def rnd_words(n, seed):
    rng = random.Random(seed)
    voc = ("Haus Garten Lampe Zug Fenster Wasser Berg Brücke Wolke Tisch Straße Hund Buch Feld Regen Licht Stern Wald Glas Stein "
           "river engine paper window market garden signal bottle forest silver bridge cloud letter").split()
    return " ".join(rng.choice(voc) for _ in range(n))

def measure(model, tag, **ctx):
    t0 = time.time()
    try:
        r = post("/api/generate", {"model": model, "keep_alive": "24h"}, timeout=1800)
        load_s = round(time.time() - t0, 1)
        err = r.get("error")
    except Exception as e:
        record(test="load", model=model, tag=tag, error=str(e)[:200], cuda_errors=runner_info()["cuda_errors"], **ctx)
        return False
    mem = gpu_mem()
    used = [m[1] for m in mem if m[1] > 300]
    info = runner_info()
    dec = []
    try:
        for _ in range(2):
            d = post("/api/generate", {"model": model, "prompt": "Write a long story about a robot.", "stream": False,
                     "options": {"num_predict": 120, "temperature": 0, "seed": 1}}, timeout=900)
            dec.append(round(d["eval_count"] / d["eval_duration"] * 1e9, 1))
        pre_prompt = rnd_words(2500, hash(tag) & 0xffff) + "\nFasse in einem Satz zusammen."
        d = post("/api/generate", {"model": model, "prompt": pre_prompt, "stream": False,
                 "options": {"num_predict": 8, "temperature": 0}}, timeout=1800)
        pre = round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 1)
        ok = True
    except Exception as e:
        pre, ok = None, False
        info["measure_error"] = str(e)[:150]
        info["cuda_errors"] = runner_info()["cuda_errors"]
    record(test="matrix", model=model, tag=tag, load_s=load_s, load_error=err, gpus_with_load=len(used),
           vram_total_mib=sum(used), vram_max_mib=max(used) if used else 0, decode=dec, prefill=pre, ok=ok, **info, **ctx)
    return ok

def phase_matrix():
    stop_all()
    # P3 Kolibri-1: spread fit with 8..12 GPUs, plus the production configuration (12 visible, greedy 1.10)
    for n in (8, 9, 10, 11, 12):
        log(f"P3 kolibri spread n={n}")
        start("ollama-gaps:kolibri-20261006", n)
        measure(KOL, f"kolibri-spread-{n}gpu", gpus=n, image="kolibri-20261006")
    log("P3 kolibri greedy 1.10 (12 visible)")
    start("ollama-gaps:kolibri-20261006", 12, force="1.10")
    measure(KOL, "kolibri-greedy110-12visible", gpus=12, image="kolibri-20261006")

def sample_mem(stop, store):
    while not stop.is_set():
        m = gpu_mem()
        store.append((time.time(), max(x[1] for x in m), max(x[2] for x in m)))
        stop.wait(20)

def phase_ladder():
    # production configuration is running (phase_matrix ends with it); ask for both needles at 25 % and 80 % depth
    for target in [int(x) for x in os.environ.get('LADDER', '32000,64000,128000,192000,245000').split(',')]:
        nparas = int(target / float(os.environ.get('TOK_PER_PARA', '72.5')))
        rng = random.Random(target)
        codes = [f"KOL-{rng.randint(1000,9999)}-{rng.choice(['ROT','BLAU','GRUEN','GELB'])}" for _ in range(2)]
        paras = [f"Absatz {i}: " + "Die Katze sitzt auf dem Dach und beobachtet die Voegel im Garten. " * 5 for i in range(nparas)]
        paras[int(nparas * 0.25)] = f"Absatz {int(nparas*0.25)}: Der erste geheime Code lautet {codes[0]}. " + paras[0][12:]
        paras[int(nparas * 0.80)] = f"Absatz {int(nparas*0.80)}: Der zweite geheime Code lautet {codes[1]}. " + paras[0][12:]
        prompt = "\n".join(paras) + "\n\nNenne den ersten und den zweiten geheimen Code. Antworte nur mit beiden Codes, durch Komma getrennt."
        stop, store = threading.Event(), []
        th = threading.Thread(target=sample_mem, args=(stop, store)); th.start()
        t0 = time.time()
        try:
            d = post("/api/chat", {"model": KOL, "stream": False, "think": True, "messages": [{"role": "user", "content": prompt}],
                     "options": {"num_predict": 1200, "temperature": 0}}, timeout=14400)
            ans = d["message"]["content"]
            record(test="ladder", target_tokens=target, prompt_tokens=d["prompt_eval_count"],
                   prefill_tok_s=round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 1),
                   decode_tok_s=round(d["eval_count"] / d["eval_duration"] * 1e9, 1), wall_s=round(time.time() - t0),
                   found=[c in ans for c in codes], answer=ans[:80],
                   vram_max_mib=max(s[1] for s in store) if store else None, temp_max=max(s[2] for s in store) if store else None)
        except Exception as e:
            record(test="ladder", target_tokens=target, error=str(e)[:200], wall_s=round(time.time() - t0), cuda_errors=runner_info()["cuda_errors"])
        finally:
            stop.set(); th.join()

def phase_soak(minutes=60):
    end = time.time() + minutes * 60
    lat, errs, n = [], 0, 0
    stop, store = threading.Event(), []
    th = threading.Thread(target=sample_mem, args=(stop, store)); th.start()
    tool = [{"type": "function", "function": {"name": "get_weather", "description": "Wetter einer Stadt",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    while time.time() < end:
        n += 1
        kind = n % 3
        if kind == 0:
            body = {"model": KOL, "stream": False, "think": True, "messages": [{"role": "user", "content": f"Was ist {n}*{n+7}? Nur die Zahl."}], "options": {"num_predict": 600, "temperature": 0}}
        elif kind == 1:
            body = {"model": KOL, "stream": False, "think": False, "messages": [{"role": "user", "content": "Wie ist das Wetter in Heidelberg?"}], "tools": tool, "options": {"num_predict": 150, "temperature": 0}}
        else:
            body = {"model": KOL, "stream": False, "think": True, "messages": [{"role": "user", "content": rnd_words(1500, n) + "\nNenne das erste Wort."}], "options": {"num_predict": 300, "temperature": 0}}
        t0 = time.time()
        try:
            d = post("/api/chat", body, timeout=1800)
            lat.append((kind, round(time.time() - t0, 1), round(d["eval_count"] / d["eval_duration"] * 1e9, 1)))
        except Exception as e:
            errs += 1
            log(f"soak error {e}")
    stop.set(); th.join()
    ds = sorted(x[2] for x in lat)
    record(test="soak", minutes=minutes, requests=n, errors=errs, decode_tok_s_min=ds[0] if ds else None,
           decode_tok_s_median=ds[len(ds)//2] if ds else None, decode_tok_s_max=ds[-1] if ds else None,
           latency_max_s=max((x[1] for x in lat), default=None), vram_max_mib=max((s[1] for s in store), default=None),
           vram_first_mib=store[0][1] if store else None, vram_last_mib=store[-1][1] if store else None,
           temp_max=max((s[2] for s in store), default=None), cuda_errors=runner_info()["cuda_errors"],
           xid=sh("dmesg 2>/dev/null | grep -ci 'NVRM: Xid'") or "n/a")

if __name__ == "__main__":
    phases = sys.argv[1:] or ["matrix", "ladder", "soak"]
    if "ladder" in phases and "matrix" not in phases:
        pass
    try:
        if "matrix" in phases: phase_matrix()
        elif "ladder" in phases:
            stop_all(); start("ollama-gaps:kolibri-20261006", 12, force="1.10")
            post("/api/generate", {"model": KOL, "keep_alive": "24h"}, timeout=1800)
        if "ladder" in phases: phase_ladder()
        if "soak" in phases: phase_soak()
    except Exception as e:
        log(f"SUITE ABORT {e!r}")
    finally:
        sh("docker rm -f ollama-m60-test >/dev/null 2>&1")
        sh(f"cd {COMPOSE} && docker compose -f docker-compose.single12.yml up -d --force-recreate >/dev/null 2>&1")
        open(f"{OUT}/DONE", "w").write(time.strftime("%F %T"))
        log("DONE (production container restored)")
