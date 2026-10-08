import json, re, subprocess, urllib.request, time
EP=[("ollama-m60-pool",11434),("ollama-m60-gpu4",11435),("ollama-m60-gpu5",11436),("ollama-m60-gpu6",11437),("ollama-m60-gpu7",11438),("ollama-m60-gpu8",11439),("ollama-m60-gpu9",11440),("ollama-m60-gpu10",11441),("ollama-m60-gpu11",11442)]
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
def get(p,path): return json.load(urllib.request.urlopen(f"http://localhost:{p}{path}",timeout=10))
for rnd in range(3):
    print(time.strftime("%H:%M:%S"), "round", rnd+1)
    for c,p in EP:
        try:
            v=get(p,"/api/version")["version"]; ps=get(p,"/api/ps")["models"]
            lg=sh(f"docker logs {c} 2>&1"); g=lambda r:(re.findall(r,lg) or [""])[-1]
            loaded=",".join(f'{m["name"].split("/")[-1][:22]}@{m["context_length"]}({round(100*m["size_vram"]/m["size"])}%)' for m in ps) or "-"
            print(f"  {c:17s} :{p} v{v} loaded={loaded:48s} loads={len(re.findall('sched.go.*loaded runners',lg))} layers={g(r'offloaded (\d+/\d+) layers') or '-'} n_batch={g(r'llama_context: n_batch\s+= (\d+)') or '-'} cuda_err={len(re.findall('CUDA error',lg))} restarts={sh(f'docker inspect {c} --format {{{{.RestartCount}}}}').strip()}")
        except Exception as e: print(f"  {c} :{p} ERROR {e}")
    if rnd<2: time.sleep(170)
