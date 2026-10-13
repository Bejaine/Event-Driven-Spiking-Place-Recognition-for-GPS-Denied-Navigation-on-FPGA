"""Run one engine configuration on all E1 modules (3 seeds x 6 modules) for one Poisson seed.

Sessions mirror the repo protocol: record (spring, 150 images, all modules),
calibrate (summer cal queries, cal modules), test (summer test queries, test modules).
Output: e3/counts/<config>/P<p>.npz with rec_S<s> (6,150,400), cal_S<s> (2,50,400),
test_S<s> (4,100,400). Skips if the output exists (resumable).
Usage: python scripts/e3_run.py CONFIG POISSON_SEED [--seeds 0 10 20]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snnsim import engine as E, inputs as I, params as P  # noqa: E402
from snnsim.configs import CONFIGS  # noqa: E402
from snnsim.modules import load_seed  # noqa: E402

CAL_OFFS = [0, 25]
TEST_OFFS = [50, 75, 100, 125]


def module_delay_seed(cfg, seed, off, p):
    # reference random delays are redrawn with every Brian2 seed; hashed delays are a fixed
    # property of the module (hardware), independent of the Poisson seed
    base = (seed * 1009 + off * 7 + 1) & 0xFFFFFFFF
    return base * 31 + p if cfg.delay_mode == "random" else base


def run(cfg_name, p, seeds, out_dir):
    cfg = CONFIGS[cfg_name]
    out = os.path.join(out_dir, cfg_name, f"P{p}.npz")
    if os.path.exists(out):
        print("exists", out); return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = {s: load_seed(s) for s in seeds}
    res, timing = {}, {}
    for session, offs, xkey, pkey in [("record", CAL_OFFS + TEST_OFFS, "record_x", "record_place"),
                                      ("cal", CAL_OFFS, "cal_x", "cal_place"),
                                      ("test", TEST_OFFS, "test_x", "test_place")]:
        nets, keys = [], []
        for si, s in enumerate(seeds):
            for off in offs:
                m = data[s]["modules"][off]
                nets.append(E.Net(m["W"], m["theta"], module_id=si * 6 + off // 25,
                                  delay_seed=module_delay_seed(cfg, s, off, p)))
                keys.append((s, off))
        # all seeds see the same images for a session only if their shuffles agree, so run per seed
        counts = {}
        t0 = time.time()
        for s in seeds:
            idx = [k for k, kk in enumerate(keys) if kk[0] == s]
            sub = [nets[k] for k in idx]
            imgs = data[s]["images"][xkey].reshape(-1, 784)
            qids = data[s]["images"][pkey]                      # place index -> same train in every seed
            trains = I.make_trains(cfg, imgs, P.steps(P.T_PRESENT, cfg.dt), p, session,
                                   [n.module_id for n in sub], qids)
            c = E.Engine(cfg, sub).run_session(trains)
            res[f"{session}_S{s}"] = c
        timing[session] = time.time() - t0
    np.savez_compressed(out, **res)
    with open(out.replace(".npz", "_timing.json"), "w") as f:
        json.dump(timing, f)
    print("wrote", out, timing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("pseed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--out", default=os.path.join(ROOT, "e3", "counts"))
    a = ap.parse_args()
    run(a.config, a.pseed, a.seeds, a.out)
