import json, re, subprocess, time, random, urllib.request, os
OUT="/home/philipp/post-deploy"; os.makedirs(OUT, exist_ok=True)
RES=open(f"{OUT}/results.jsonl","a",buffering=1)
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
def post(port,path,body,to=1800):
    r=urllib.request.Request(f"http://localhost:{port}{path}",json.dumps(body).encode(),{"Content-Type":"application/json"}); return json.load(urllib.request.urlopen(r,timeout=to))
def get(port,path): return json.load(urllib.request.urlopen(f"http://localhost:{port}{path}",timeout=60))
def test(port,cont,model,tag):
    t0=time.time()
    try:
        post(port,"/api/generate",{"model":model,"keep_alive":"24h"}); load=round(time.time()-t0,1)
        dec=[]
        for _ in range(3):
            d=post(port,"/api/generate",{"model":model,"prompt":"Write a long story about a robot.","stream":False,"options":{"num_predict":120,"temperature":0,"seed":1}}); dec.append(round(d["eval_count"]/d["eval_duration"]*1e9,1))
        rng=random.Random(len(tag)); w="Haus Garten Lampe Zug Fenster Wasser Berg river engine paper window market garden".split()
        d=post(port,"/api/generate",{"model":model,"prompt":" ".join(rng.choice(w) for _ in range(2500))+"\nSummarize in one sentence.","stream":False,"options":{"num_predict":8,"temperature":0}}); pre=round(d["prompt_eval_count"]/d["prompt_eval_duration"]*1e9,1)
        c=post(port,"/api/chat",{"model":model,"stream":False,"think":False,"messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"options":{"num_predict":400,"temperature":0}})
        lg=sh(f"docker logs {cont} 2>&1"); g=lambda p:(re.findall(p,lg) or [""])[-1]
        ps=get(port,"/api/ps")["models"]
        rec=dict(test=tag,model=model,ok=True,load_s=load,decode=dec,prefill=pre,layers=g(r"offloaded (\d+/\d+) layers"),n_ctx=g(r"llama_context: n_ctx\s+= (\d+)"),n_batch=g(r"llama_context: n_batch\s+= (\d+)"),
                 host_buf=g(r"CUDA_Host compute buffer size\s+=\s+([\d.]+)"),spec="--spec" in sh(f"docker exec {cont} sh -c 'ps aux | grep llama-server | grep -v grep'"),cuda_errors=len(re.findall("CUDA error",lg)),
                 size_vram=[m["size_vram"] for m in ps],size=[m["size"] for m in ps],answer=(c["message"]["content"] or "")[-60:].replace("\n"," "))
    except Exception as e: rec=dict(test=tag,model=model,ok=False,error=str(e)[:200])
    rec["ts"]=time.strftime("%FT%T"); RES.write(json.dumps(rec)+"\n")
for port,cont in ((11436,"ollama-m10"),(11442,"ollama-m60-guard")):
    tags=[m["name"] for m in get(port,"/api/tags")["models"]]
    RES.write(json.dumps(dict(test=f"tags-{port}",count=len(tags),version=get(port,"/api/version"),sample=tags[:3]))+"\n")
test(11436,"ollama-m10","qwen3.6:35b","m10-qwen3.6-35b")
test(11442,"ollama-m60-guard","llama-guard3:8b","guard-llama-guard3-8b")
test(11442,"ollama-m60-guard","qwen3.5:9b","guard-qwen3.5-9b")
open(f"{OUT}/DONE","w").write("done")
