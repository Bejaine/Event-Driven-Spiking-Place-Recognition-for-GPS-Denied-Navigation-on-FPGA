"""E1 evaluation with the repo's own scripts (calibrate -> threshold -> test -> ensemble -> SeqSLAM).

For Brian2 run set P (P=0: the training run dirs e1/S<seed>; P>0: noise-band re-runs
e1/S<seed>_P<P>), executes, with the run dir as CWD:
  modular_snn/modular_snn_evaluation.py --process_mode calibrate   (13 hyperactive thresholds)
  modular_snn/modular_snn_evaluation.py --process_mode test
  ens_seq/process_ensembles.py  (3 seeds)
  ens_seq/process_seqmatch.py   (per seed and for the ensemble; Lseq = 2 4 10, plus 4 for E3)
and writes results/e1_brian2_P<P>.json with every R@N read back from the .npy files the
repo scripts save. Threshold choice: best calibration R@1, ties -> smallest threshold.
Usage: python scripts/e1_evaluate.py P
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "envs", "vprsnn", "bin", "python")
REPO = os.path.join(ROOT, "VPRSNN")
SEEDS = [0, 10, 20]
NCAL, NTEST, NL, EPOCHS = 50, 100, 25, 30
NQ = NCAL + NTEST
THRESHOLDS = [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
ENV = dict(os.environ, WANDB_MODE="disabled", MPLBACKEND="Agg")


def sh(cmd, cwd, log):
    with open(log, "a") as f:
        f.write(f"\n$ (cd {cwd}) {' '.join(cmd)}\n"); f.flush()
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=ENV, check=True)


def recall(path):
    d = np.load(path, allow_pickle=True).item()
    return {int(k): float(v) for k, v in d.items()}


def main(P):
    rundir = (lambda s: f"S{s}") if P == 0 else (lambda s: f"S{s}_P{P}")
    log = os.path.join(ROOT, "logs", f"e1_evaluate_P{P}.log")
    common = ["--dataset", "nordland", "--num_labels", str(NL), "--num_cal_labels", str(NCAL),
              "--num_test_labels", str(NTEST), "--skip", "8", "--folder_id", "NRD_SFS",
              "--n_e", "400", "--epochs", str(EPOCHS)]
    out = {"P": P, "seeds": {}, "args": common}
    thr = {}
    for s in SEEDS:
        cwd = os.path.join(ROOT, "e1", rundir(s))
        # repo CLI is broken (duplicate --num_query_imgs); call its main() via a wrapper
        ev = os.path.join(ROOT, "scripts", "run_modular_eval.py")
        sh([PY, ev, "calibrate", str(s), "0", str(NCAL), str(NTEST), str(EPOCHS)], cwd, log)
        cal = {}
        for th in THRESHOLDS:
            p = os.path.join(cwd, f"outputs/outputs_ne{400 * NCAL // NL}_L{NCAL}_offset{NCAL - NL}_M2/standard/epoch{EPOCHS}_T{NQ}_T{th}/recallAtN_SNN.npy")
            cal[th] = recall(p)[1] if os.path.exists(p) else None
        best = None
        for th in THRESHOLDS:
            if cal[th] is not None and (best is None or cal[th] > cal[best]):
                best = th
        thr[s] = best
        sh([PY, ev, "test", str(s), str(NCAL), str(NCAL), str(NTEST), str(EPOCHS)], cwd, log)
        tdir = f"outputs/outputs_ne{400 * NTEST // NL}_L{NTEST}_offset{NQ - NL}_M2/standard/epoch{EPOCHS}_T{NQ}_T{{}}"
        test_all = {}
        for th in THRESHOLDS:
            p = os.path.join(cwd, tdir.format(th), "recallAtN_SNN.npy")
            test_all[th] = recall(p) if os.path.exists(p) else None
        out["seeds"][s] = {"cal_R1_by_threshold": cal, "threshold": best, "test": test_all[best],
                           "test_by_threshold": {k: (v[1] if v else None) for k, v in test_all.items()},
                           "summed_rates": os.path.join(cwd, tdir.format(best), "summed_rates.npy")}
    # ensemble + seqmatch: a main folder per run set, holding the shuffled indices
    main_dir = os.path.join(ROOT, "e1", f"ens_P{P}")
    os.makedirs(main_dir, exist_ok=True)
    for s in SEEDS:
        shutil.copy(os.path.join(ROOT, "e1", rundir(s), f"shuffled_indices_L{NQ}_S{s}.npy"), main_dir)
    data_fmt = "../" + rundir("{}") + f"/outputs/outputs_ne{400 * NTEST // NL}_L{NTEST}_offset{NQ - NL}_M2/standard/epoch{{}}_T{NQ}_T{{}}"
    ensm = os.path.join(REPO, "ens_seq", "process_ensembles.py")
    sh([PY, ensm, "--SNN_data_path", data_fmt, "--mainfolder_path", main_dir, "--seeds"] + [str(s) for s in SEEDS]
       + ["--epochs"] + [str(EPOCHS)] * 3 + ["--thresholds"] + [str(thr[s]) for s in SEEDS]
       + ["--num_query_imgs", str(NQ), "--num_cal_labels", str(NCAL)], ROOT, log)
    ens_dir = os.path.join(main_dir, f"merged_results_ens_SNN_S{len(SEEDS)}")
    out["ensemble"] = {"SL1": recall(os.path.join(ens_dir, f"recallAtN_ens_{len(SEEDS)}_SNNs_SL1.npy"))}
    seqm = os.path.join(REPO, "ens_seq", "process_seqmatch.py")
    lens = ["2", "4", "10"]
    for s in SEEDS:
        sh([PY, seqm, "--SNN_data_path", data_fmt + "/", "--mainfolder_path", main_dir, "--seed", str(s),
            "--epochs", str(EPOCHS), "--thresholds", str(thr[s]), "--num_query_imgs", str(NQ),
            "--num_cal_labels", str(NCAL), "--seq_len_list"] + lens, ROOT, log)
        d = os.path.join(main_dir, f"merged_results_seq_SNN_S{s}")
        out["seeds"][s]["seq"] = {L: recall(os.path.join(d, f"recallAtN_SNN{s}_SL{L}.npy")) for L in [1] + [int(x) for x in lens]}
    sh([PY, seqm, "--SNN_data_path", f"merged_results_ens_SNN_S{len(SEEDS)}/", "--mainfolder_path", main_dir,
        "--seed", str(len(SEEDS)), "--num_query_imgs", str(NQ), "--num_cal_labels", str(NCAL),
        "--use_ensemble", "True", "--seq_len_list"] + lens, ROOT, log)
    out["ensemble"]["seq"] = {L: recall(os.path.join(ens_dir, f"recallAtN_SNN{len(SEEDS)}_SL{L}.npy")) for L in [1] + [int(x) for x in lens]}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"e1_brian2_P{P}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", path)
    for s in SEEDS:
        print(s, "thr", thr[s], "R@1", out["seeds"][s]["test"][1], "R@5", out["seeds"][s]["test"][5])
    print("ensemble R@1", out["ensemble"]["SL1"][1])


if __name__ == "__main__":
    main(int(sys.argv[1]))
