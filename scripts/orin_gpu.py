"""J4/J5 (Jetson Orin Nano): GPU throughput, latency and power.

  cuda  : golden_cuda/snn_cuda (our CUDA port of the fixed-point golden model, bit-exact)
          bench at dt = 0.1 and 0.5 ms, then sustained `loop` windows (~60 s) with the power
          sampler: batch 4 (one 4-module query at a time) and batch 96 (saturated GPU),
          bracketed by idle windows. -> results/orin_cuda.json
  torch : the DGX tooling engine (snnsim/engine_torch.py, float32, scatter-bound), same
          workload as scripts/e5_gpu.py (12 E1 test modules x their 100 test queries), with
          the power sampler running; out-of-memory batches are recorded and skipped.
          -> results/orin_gpu.json
Usage: python scripts/orin_gpu.py cuda|torch
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from orin_cpu_bench import cool_down, idle_window, power_summary, start_sampler  # noqa: E402

CUDA = os.path.join(ROOT, "golden_cuda", "snn_cuda")
GV = os.path.join(ROOT, "golden_vectors")


def cuda_main():
    out = {"engine": "golden_cuda/snn_cuda (fixed-point, bit-exact with golden_c; one block per module run)",
           "uptime_before": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}
    for dt_us in (100, 500):
        cool_down()
        jf = os.path.join(ROOT, "cache", f"cuda_bench_dt{dt_us}.json")
        smp, csv = start_sampler(f"j4_cuda_bench_dt{dt_us}", 900)
        time.sleep(1.0)
        t0 = time.time()
        subprocess.run(["taskset", "-c", "3", CUDA, "bench", GV, str(dt_us), "25", jf], check=True)
        t1 = time.time()
        smp.terminate(); smp.wait()
        r = json.load(open(jf))
        r["power_during_bench"] = power_summary(csv, t0, t1)
        r["power_csv"] = os.path.relpath(csv, ROOT)
        out[f"bench_dt{dt_us}"] = r
        print("bench", dt_us, r["single_run"], flush=True)
    win = {"idle_1": idle_window("j5_cuda_idle_1")}
    for name, batch in (("loop_batch4", 4), ("loop_batch96", 96)):
        cool_down()
        smp, csv = start_sampler(f"j5_cuda_{name}", 200)
        time.sleep(1.0)
        r = subprocess.run(["taskset", "-c", "3", CUDA, "loop", GV, "100", str(batch), "60"],
                           capture_output=True, text=True, check=True)
        smp.terminate(); smp.wait()
        res = json.loads(r.stdout)
        res["power"] = power_summary(csv, res["t_start"], res["t_end"])
        res["power_csv"] = os.path.relpath(csv, ROOT)
        win[name] = res
        print(name, {k: res[k] for k in ("batch", "module_runs_per_s", "wall_s")}, flush=True)
        win[f"idle_after_{name}"] = idle_window(f"j5_cuda_idle_after_{name}")
    out["power"] = win
    with open(os.path.join(ROOT, "results", "orin_cuda.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/orin_cuda.json")


def torch_main():
    import numpy as np
    import torch
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
    allimg = np.concatenate(imgs)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "label": "tooling engine, float32, scatter-bound (snnsim/engine_torch.py), not an optimised deployment",
           "workload": "scripts/e5_gpu.py: 12 E1 test modules x 100 test queries, hard reset, random 0.1 ms delays",
           "runs": [], "failed": []}
    for dtype, name in ((torch.float32, "float32"), (torch.float16, "float16")):
        eng = TorchEngine(np.stack(W), np.stack(th), np.stack(D), dtype=dtype)
        eng.run_hard(np.arange(12) % 12, allimg[:12], seed=1)   # warm-up / allocator
        for N in [600, 1200, 2400, 4800]:
            mi = np.arange(N) % 12
            im = allimg[np.arange(N) % len(allimg)]
            cool_down()
            smp, csv = start_sampler(f"j4_torch_{name}_{N}", 1800)
            time.sleep(1.0)
            try:
                torch.cuda.synchronize(); t0 = time.time()
                c = eng.run_hard(mi, im, seed=1)
                torch.cuda.synchronize(); el = time.time() - t0
            except torch.OutOfMemoryError as e:
                smp.terminate(); smp.wait()
                res["failed"].append({"dtype": name, "batch": N, "error": "OutOfMemoryError: " + str(e)[:300]})
                print("OOM", name, N, flush=True)
                torch.cuda.empty_cache()
                continue
            smp.terminate(); smp.wait()
            r = {"dtype": name, "batch": N, "wall_s": el, "module_runs_per_s": N / el,
                 "mean_e_spikes": float(c.sum(1).mean()), "max_mem_bytes": torch.cuda.max_memory_allocated(),
                 "power": power_summary(csv, t0, t0 + el), "power_csv": os.path.relpath(csv, ROOT)}
            res["runs"].append(r)
            print({k: r[k] for k in ("dtype", "batch", "wall_s", "module_runs_per_s", "mean_e_spikes")}, flush=True)
            torch.cuda.reset_peak_memory_stats()
        del eng
        torch.cuda.empty_cache()
    f32 = [r["module_runs_per_s"] for r in res["runs"] if r["dtype"] == "float32"]
    res["best_module_runs_per_s"] = max(f32) if f32 else None
    res["idle_after"] = idle_window("j5_torch_idle_after")
    with open(os.path.join(ROOT, "results", "orin_gpu.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote results/orin_gpu.json")


if __name__ == "__main__":
    {"cuda": cuda_main, "torch": torch_main}[sys.argv[1]]()
