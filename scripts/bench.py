#!/usr/bin/env python3
"""Load models into Ollama, report memory placement and generation speed."""
import json, subprocess, sys, urllib.request

HOST = "http://127.0.0.1:11434"
PROMPT = ("You are a planning agent. A user wants to migrate a Flask app to FastAPI. "
          "List the steps, the risks, and which tools you would call. Be detailed.")


def post(path, body):
    req = urllib.request.Request(HOST + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.load(r)


def unload_all():
    for m in get("/api/ps").get("models", []):
        post("/api/generate", {"model": m["name"], "keep_alive": 0})


def gen(model, ctx):
    r = post("/api/generate", {
        "model": model, "prompt": PROMPT, "stream": False, "think": False,
        "keep_alive": "30m",
        "options": {"num_ctx": ctx, "num_predict": 300, "temperature": 0},
    })
    tps = r["eval_count"] / (r["eval_duration"] / 1e9)
    pps = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
    return (f"{model:24} ctx={ctx:<6} load={r['load_duration']/1e9:5.1f}s "
            f"prompt={pps:7.1f} tok/s  gen={tps:6.1f} tok/s")


def embed(model):
    r = post("/api/embed", {"model": model, "input": [PROMPT] * 16, "keep_alive": "30m"})
    return f"{model:24} 16 embeddings in {r['total_duration']/1e9:.2f}s (dim={len(r['embeddings'][0])})"


def placement():
    print("\n  loaded models:")
    for m in get("/api/ps")["models"]:
        size, vram = m["size"], m["size_vram"]
        print(f"    {m['name']:24} total={size/2**30:5.1f} GiB  on GPU={vram/2**30:5.1f} GiB "
              f"({100*vram/size:3.0f}%)  ctx={m.get('context_length', '?')}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"  nvidia-smi used/total: {smi}\n")


def scenario(name, steps):
    print(f"### {name}")
    unload_all()
    for kind, model, *ctx in steps:
        print("  " + (embed(model) if kind == "embed" else gen(model, ctx[0])))
    placement()
    # second pass: all resident, measures speed with no reload
    for kind, model, *ctx in steps:
        if kind == "gen":
            print("  warm " + gen(model, ctx[0]))
    print()


if __name__ == "__main__":
    which = sys.argv[1:] or ["trio", "moe"]
    if "trio" in which:
        scenario("A: agent trio, all resident", [
            ("gen", "qwen3.5:latest", 32768),
            ("gen", "qwen3.5:4b", 32768),
            ("embed", "qwen3-embedding:0.6b"),
        ])
    if "moe" in which:
        scenario("B: qwen3.6:35b-a3b alone", [("gen", "qwen3.6:35b-a3b", 32768)])
        scenario("B2: qwen3.6:35b-a3b alone, 8K ctx", [("gen", "qwen3.6:35b-a3b", 8192)])
    if "big64k" in which:
        # What bin/claude-local runs: the 35B with a 64K window. Watch the GPU share in `placement`.
        scenario("C: qwen3.6:35b-a3b, 64K ctx (claude-local)", [("gen", "qwen3.6:35b-a3b", 65536)])
    unload_all()
