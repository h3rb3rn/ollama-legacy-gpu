import json, re, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
A = [("ollama-m60-pool",11434,"01-king","qwen3.6:35b",262144),("ollama-m60-gpu4",11435,"02-explorer","qwen3.5:4b",262144),
 ("ollama-m60-gpu5",11436,"03-librarian","granite4.2:3b",131072),("ollama-m60-gpu6",11437,"04-artisan","granite4.2:3b",131072),
 ("ollama-m60-gpu7",11438,"05-interpreter","gemma3:4b",131072),("ollama-m60-gpu8",11439,"06-operator","nemotron-3-nano:4b",262144),
 ("ollama-m60-gpu9",11440,"07-methodologist","huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF:latest",262144),
 ("ollama-m60-gpu10",11441,"08-logician","hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M",262144),
 ("ollama-m60-gpu11",11442,"09-chronicler","hf.co/webAI-Official/TwIL-LM3-Pro:Q4_K_M",131072)]
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
def post(port,path,body,to=1800):
    r=urllib.request.Request(f"http://localhost:{port}{path}",json.dumps(body).encode(),{"Content-Type":"application/json"}); return json.load(urllib.request.urlopen(r,timeout=to))
def one(a):
    cont,port,agent,model,ctx=a; t0=time.time()
    try:
        post(port,"/api/generate",{"model":model,"keep_alive":"24h","options":{"num_ctx":ctx}}); load=round(time.time()-t0,1)
        ps=json.load(urllib.request.urlopen(f"http://localhost:{port}/api/ps",timeout=30))["models"][0]
        dec=[]
        for _ in range(2):
            d=post(port,"/api/generate",{"model":model,"prompt":"Write a short story about a robot.","stream":False,"options":{"num_ctx":ctx,"num_predict":100,"temperature":0,"seed":1}}); dec.append(round(d["eval_count"]/d["eval_duration"]*1e9,1))
        lg=sh(f"docker logs {cont} 2>&1"); g=lambda p:(re.findall(p,lg) or [""])[-1]
        ps2=json.load(urllib.request.urlopen(f"http://localhost:{port}/api/ps",timeout=30))["models"][0]
        return dict(agent=agent,port=port,model=model,ctx=ctx,ok=True,load_s=load,decode=dec,gpu_pct=round(100*ps2["size_vram"]/ps2["size"],1),size_gb=round(ps2["size"]/2**30,2),
                    layers=g(r"offloaded (\d+/\d+) layers"),n_ctx=g(r"llama_context: n_ctx\s+= (\d+)"),n_batch=g(r"llama_context: n_batch\s+= (\d+)"),
                    host_buf=g(r"CUDA_Host compute buffer size\s+=\s+([\d.]+)"),cuda_err=len(re.findall("CUDA error",lg)),ps_ctx=ps2.get("context_length"))
    except Exception as e: return dict(agent=agent,port=port,model=model,ctx=ctx,ok=False,error=str(e)[:160],cuda_err=len(re.findall("CUDA error",sh(f"docker logs {cont} 2>&1"))))
with ThreadPoolExecutor(9) as ex: res=list(ex.map(one,A))
for r in res: print(json.dumps(r,ensure_ascii=False))
