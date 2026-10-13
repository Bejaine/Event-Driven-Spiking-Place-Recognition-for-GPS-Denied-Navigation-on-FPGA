"""E5 (tooling only): throughput of the batched GPU float engine in module-runs per second.

12 E1 test modules (3 seeds x 4) x their 100 test queries, hard-reset module runs,
random 0.1 ms delays, float32 on the GB10 GPU. Writes results/e5_gpu.json.
"""
import json, os, sys, time
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snnsim.engine_torch import TorchEngine
from snnsim.modules import load_seed

W, th, imgs, D = [], [], [], []
rng = np.random.default_rng(0)
for s in [0, 10, 20]:
    d = load_seed(s)
    for o in [50, 75, 100, 125]:
        W.append(d["modules"][o]["W"]); th.append(d["modules"][o]["theta"])
        D.append(np.floor(rng.random((784, 400)) * 100 + 0.5).astype(np.int64))
    imgs.append(d["images"]["test_x"].reshape(-1, 784))
eng = TorchEngine(np.stack(W), np.stack(th), np.stack(D), dtype=torch.float32)
allimg = np.concatenate(imgs)            # 300 queries
res = {"device": torch.cuda.get_device_name(0), "dtype": "float32", "runs": []}
for N in [600, 1200, 2400, 4800]:
    mi = np.arange(N) % 12
    im = allimg[np.arange(N) % len(allimg)]
    torch.cuda.synchronize(); t = time.time()
    c = eng.run_hard(mi, im, seed=1)
    torch.cuda.synchronize(); el = time.time() - t
    res["runs"].append({"batch": N, "wall_s": el, "module_runs_per_s": N / el, "mean_e_spikes": float(c.sum(1).mean())})
    print(res["runs"][-1], flush=True)
res["best_module_runs_per_s"] = max(r["module_runs_per_s"] for r in res["runs"])
json.dump(res, open(os.path.join(ROOT, "results", "e5_gpu.json"), "w"), indent=1)
