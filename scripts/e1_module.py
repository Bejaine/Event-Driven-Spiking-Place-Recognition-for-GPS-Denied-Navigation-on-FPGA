"""Run one VPRSNN Modular-SNN module through the repo's own pipeline, with timing.

Thin wrapper around VPRSNN modular_snn/one_snn_module_processing.py
(https://github.com/QVPR/VPRSNN, MIT License, (c) 2023 Hussaini, Milford, Fischer).
It does not change the model; it only
  * builds the same argparse namespace the repo's CLI would,
  * times every Brian2 `Network.run` call (to separate per-image inference time from
    network construction / code generation), and every snn_model.main call,
  * optionally re-seeds Brian2's RNG for inference (Poisson trains + delays) with
    `--poisson_seed` != 0, used to measure Brian2's own seed-to-seed noise band.
    The image shuffling still uses --seed, so module/place assignment is unchanged.

Must be run with the per-seed run directory as CWD (the repo uses relative paths:
./outputs, ./weights, ./random, ./dataset_imagenames, ./../data).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
os.environ.setdefault("WANDB_MODE", "disabled")

import brian2 as b2  # noqa: E402

TIMING = {"runs": []}


def _patch_network_run():
    orig = b2.Network.run

    def timed_run(self, duration, *a, **k):
        # Brian2 resolves identifiers in the caller's frame: skip this wrapper frame
        k["level"] = k.get("level", 0) + 1
        t0 = time.perf_counter()
        r = orig(self, duration, *a, **k)
        TIMING["runs"].append((float(duration), time.perf_counter() - t0))
        return r
    b2.Network.run = timed_run


def _patch_seed(poisson_seed):
    """Re-seed Brian2 for inference only (Poisson trains + delays), keeping --seed shuffling.

    snn_model.main calls b2.seed(seed) and THEN get_shuffled_indices(), which re-seeds
    numpy's global RNG with `seed` (Brian2's cython rand() draws from it). So the re-seed
    must happen right after get_shuffled_indices returns.
    """
    if poisson_seed == 0:
        return
    import tools.snn_model_utils as smu
    orig = smu.get_shuffled_indices

    def wrapped(num_test_labels, num_cal_labels, shuffled=True, seed=0):
        r = orig(num_test_labels, num_cal_labels, shuffled=shuffled, seed=seed)
        b2.seed(int(seed) * 1000 + poisson_seed)
        return r
    smu.get_shuffled_indices = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--mode", choices=["train", "record", "test", "calibrate"], required=True)
    ap.add_argument("--num_labels", type=int, default=25)
    ap.add_argument("--num_cal_labels", type=int, default=50)
    ap.add_argument("--num_test_labels", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--update_interval", type=int, default=250)
    ap.add_argument("--poisson_seed", type=int, default=0)
    ap.add_argument("--timing_out", type=str, required=True)
    a = ap.parse_args()

    _patch_network_run()
    _patch_seed(a.poisson_seed)
    from modular_snn import one_snn_module_processing as osp
    import non_modular_snn.snn_model as sm

    calls = []

    def timed_main(args):
        t0 = time.perf_counter()
        n0 = len(TIMING["runs"])
        sm.main(args)
        calls.append({"process_mode": args.process_mode, "ad_path_test": args.ad_path_test,
                      "num_test_labels": args.num_test_labels, "wall_s": time.perf_counter() - t0,
                      "net_runs": TIMING["runs"][n0:]})
    osp.snn_model_main = timed_main

    args = argparse.Namespace(
        dataset="nordland", num_labels=a.num_labels, num_cal_labels=a.num_cal_labels,
        num_test_labels=a.num_test_labels, num_query_imgs=a.num_cal_labels + a.num_test_labels,
        tc_ge=1.0, tc_gi=0.5, intensity=4, use_weighted_assignments=False, shuffled=True,
        skip=8, offset_after_skip=a.offset, folder_id="NRD_SFS", update_interval=a.update_interval,
        epochs=a.epochs, n_e=400, threshold_i=0, seed=a.seed, ad_path_test="_test_E{}",
        ad_path=f"_offset{a.offset}", multi_path="epoch{}_T{}_T{}", process_mode=a.mode)
    t0 = time.perf_counter()
    osp.main(args)
    total = time.perf_counter() - t0
    sys.stdout = sys.__stdout__
    with open(a.timing_out, "w") as f:
        json.dump({"args": vars(a), "total_wall_s": total, "calls": calls,
                   "cpu": os.sched_getaffinity(0) and sorted(os.sched_getaffinity(0))}, f)
    print("done", total)


if __name__ == "__main__":
    main()
