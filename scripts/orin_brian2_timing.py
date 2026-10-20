"""J3 (Jetson Orin Nano): Brian2 per-image latency of one E1 test module through VPRSNN's own
test-mode network, without the Nordland images on disk.

Runs scripts/e1_module.py --mode test unchanged, except that VPRSNN's processImageDataset is
replaced by a loader that returns the pre-processed test images saved on the DGX
(e1/S<seed>/images.npz test_x / test_y). prepare_images.py made those arrays with the same
processImageDataset call that test mode issues (num_test_labels=100, offset_after_skip=50),
so the Brian2 network, PoissonGroup input, seeds and timing hooks are all VPRSNN's.
The output resultPopVecs is compared with the DGX one in e1/S<seed> as a workload check.

Run inside a copy of the run dir (scripts/e1_setup_rundir.sh 0 9 -> e1/S0_P9), pinned:
  cd e1/S0_P9 && taskset -c 3 python ../../scripts/orin_brian2_timing.py run 0 50 OUT.json
  python scripts/orin_brian2_timing.py summary OUT.json [label]  -> per-image medians
"""
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(seed, offset, out):
    sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import non_modular_snn.snn_model as sm
    imgs = np.load(os.path.join(ROOT, "e1", f"S{seed}", "images.npz"))

    def from_npz(path, type, imWidth, imHeight, num_patches=7, num_labels=25, num_test_labels=2700,
                 num_query_imgs=3300, skip=0, offset_after_skip=0, shuffled_indices=()):
        assert type == "test" and num_test_labels == len(imgs["test_x"]) and offset_after_skip == 50, \
            (type, num_test_labels, offset_after_skip)
        assert np.array_equal(np.asarray(shuffled_indices), imgs["shuffled_indices"])
        print(f"[orin] test images from images.npz ({len(imgs['test_x'])}), not ./../data", flush=True)
        return {"x": imgs["test_x"], "y": imgs["test_y"].reshape(-1, 1), "rows": imWidth, "cols": imHeight}
    sm.processImageDataset = from_npz

    import e1_module
    sys.argv = ["e1_module.py", "--seed", str(seed), "--offset", str(offset), "--mode", "test", "--timing_out", out]
    e1_module.main()
    # workload check against the DGX Brian2 output for the same module
    rel = os.path.join("outputs", f"outputs_ne400_L25_offset{offset}", "resultPopVecs100_test_E30.npy")
    mine = np.load(rel)
    dgx = np.load(os.path.join(ROOT, "e1", f"S{seed}", rel))
    t = json.load(open(out))
    t["result_vs_dgx"] = {"identical": bool(np.array_equal(mine, dgx)), "max_abs_diff": float(np.abs(mine - dgx).max()),
                          "orin_total_spikes": float(mine.sum()), "dgx_total_spikes": float(dgx.sum())}
    json.dump(t, open(out, "w"))
    print("result vs DGX:", t["result_vs_dgx"], flush=True)


def summary(path):
    t = json.load(open(path))
    runs = [r for c in t["calls"] for r in c["net_runs"]]
    pres = [s for d, s in runs if abs(d - 0.35) < 1e-9]
    rest = [s for d, s in runs if abs(d - 0.15) < 1e-9]
    n = min(len(pres), len(rest))
    per = [pres[i] + rest[i] for i in range(n)]
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    return {"cpu": t.get("cpu"), "n_images": n, "median_s_per_image": statistics.median(per),
            "p10": q(per, 0.1), "p90": q(per, 0.9),
            "present_350ms_median_s": statistics.median(pres), "rest_150ms_median_s": statistics.median(rest),
            "test_call_wall_s": sum(c["wall_s"] for c in t["calls"]), "total_wall_s": t["total_wall_s"],
            "result_vs_dgx": t.get("result_vs_dgx"),
            "other_net_runs": [(d, s) for d, s in runs if abs(d - 0.35) > 1e-9 and abs(d - 0.15) > 1e-9]}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), os.path.abspath(sys.argv[4]))
    else:
        print(json.dumps(summary(sys.argv[2]), indent=1))
