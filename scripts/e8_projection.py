"""E8 (DERIVED): cost of training a 500-place route x 3 ensemble members on this machine,
from the MEASURED E1 Brian2 training wall times (30 epochs, 25 places x 2 traverses per module).
Assumptions: 500 places / 25 = 20 modules per member, 60 module trainings; one process per core,
18 concurrent (as in E1); per-module time scales linearly with epochs and reference traverses."""
import glob, json, math, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
x925 = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}
t = {"x925": [], "a725": []}
for f in glob.glob(os.path.join(ROOT, "e1", "S*", "timing", "O*_train.json")):
    if "_P" in f.split(os.sep)[-3]:
        continue
    d = json.load(open(f))
    c = [c for c in d["calls"] if c["process_mode"] == "train"][0]
    t["x925" if d["cpu"][0] in x925 else "a725"].append(c["wall_s"])
allt = t["x925"] + t["a725"]
n = 60
res = {"inputs": {k: {"median_s": float(np.median(v)), "max_s": float(max(v)), "n": len(v)} for k, v in t.items()},
       "module_trainings": n, "epochs": 30, "concurrency": 18}
for ep in (30, 60):
    s = ep / 30
    res[f"epochs{ep}"] = {"core_hours": n * float(np.median(allt)) * s / 3600,
                          "wall_h_18_procs": math.ceil(n / 18) * max(allt) * s / 3600,
                          "wall_h_18_procs_x925_only_estimate": math.ceil(n / 10) * max(t["x925"]) * s / 3600}
json.dump(res, open(os.path.join(ROOT, "results", "e8_training_projection.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
