"""E2/E3 aggregation: score Brian2 runs and every engine configuration identically.

Sources of spike-count matrices:
  brian2 P<p> : e1/S<seed>[ _P<p>]/outputs/outputs_ne400_L25_offset<o>/resultPopVecs*.npy
  engine cfg  : e3/counts/<cfg>/P<p>.npz   (scripts/e3_run.py)
For each (source, Poisson seed, model seed): hyperactive threshold calibrated on the
calibration modules (best R@1, ties -> smallest), then test R@1/R@5, SeqSLAM (Lseq 2,4,10),
and the 3-seed ensemble (+ SeqSLAM L4). Scoring = VPRSNN functions via snnsim/evaluate.py.
E2 fidelity: per-neuron mean test spike counts (over the 100 test queries) compared
between sources (Pearson r and mean |diff|), for every pair of Poisson seeds.
Writes results/e3_summary.json.
"""
import glob
import json
import os
import sys
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snnsim import evaluate as EV  # noqa: E402
from snnsim.modules import OFFSETS  # noqa: E402

SEEDS = [0, 10, 20]
CAL, TEST = [0, 25], [50, 75, 100, 125]
NCAL, NTEST, EPOCHS = 50, 100, 30


def assignments(seed):
    run = os.path.join(ROOT, "e1", f"S{seed}")
    return {o: np.load(os.path.join(run, "outputs", f"outputs_ne400_L25_offset{o}", f"assignments_L25_S8_O{o}.npy"))
            for o in OFFSETS}


def shuffled(seed):
    return np.load(os.path.join(ROOT, "e1", f"S{seed}", f"shuffled_indices_L{NCAL + NTEST}_S{seed}.npy"))


def brian2_counts(seed, p):
    run = os.path.join(ROOT, "e1", f"S{seed}" if p == 0 else f"S{seed}_P{p}")
    d = lambda o: os.path.join(run, "outputs", f"outputs_ne400_L25_offset{o}")
    if not all(os.path.exists(os.path.join(d(o), f"resultPopVecs{NCAL + NTEST}_record_E{EPOCHS}.npy")) for o in OFFSETS):
        return None
    rec = {o: np.load(os.path.join(d(o), f"resultPopVecs{NCAL + NTEST}_record_E{EPOCHS}.npy")) for o in OFFSETS}
    cal = {o: np.load(os.path.join(d(o), f"resultPopVecs{NCAL}_test_E{EPOCHS}.npy")) for o in CAL}
    tst = {o: np.load(os.path.join(d(o), f"resultPopVecs{NTEST}_test_E{EPOCHS}.npy")) for o in TEST}
    return rec, cal, tst


def engine_counts(cfg, seed, p):
    f = os.path.join(ROOT, "e3", "counts", cfg, f"P{p}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    r, c, t = z[f"record_S{seed}"], z[f"cal_S{seed}"], z[f"test_S{seed}"]
    rec = {o: r[i] for i, o in enumerate(CAL + TEST)}
    cal = {o: c[i] for i, o in enumerate(CAL)}
    tst = {o: t[i] for i, o in enumerate(TEST)}
    return rec, cal, tst


def score_seed(counts, seed, A, SH):
    rec, cal, tst = counts
    th, table = EV.calibrate_threshold([rec[o] for o in CAL], [cal[o] for o in CAL], [A[o] for o in CAL], SH, NCAL)
    r, sr = EV.evaluate_split([rec[o] for o in TEST], [tst[o] for o in TEST], [A[o] for o in TEST], SH, NCAL, "test", th)
    from non_modular_snn.snn_model_evaluation import get_unshuffled_results
    sru = get_unshuffled_results(sr, SH, NCAL)
    seq = {L: EV.seqslam_recall(sru, L)[1] for L in (2, 4, 10)}
    return {"threshold": th, "cal_table": table, "R1": r[1], "R5": r[5], "seq": seq}, sr


def score_source(get, ps):
    A = {s: assignments(s) for s in SEEDS}
    SH = {s: shuffled(s) for s in SEEDS}
    out = {}
    for p in ps:
        per, srs = {}, []
        for s in SEEDS:
            c = get(s, p)
            if c is None:
                break
            per[s], sr = score_seed(c, s, A[s], SH[s])
            srs.append(sr)
        if len(per) < len(SEEDS):
            continue
        ens, tot = EV.ensemble_recall(srs, [SH[s] for s in SEEDS], NCAL)
        out[p] = {"seeds": per, "ens_R1": ens[1], "ens_R5": ens[5],
                  "ens_seq": {L: EV.seqslam_recall(tot, L)[1] for L in (2, 4, 10)}}
    return out


def neuron_means(get, p):
    out = []
    for s in SEEDS:
        c = get(s, p)
        if c is None:
            return None
        out.append(np.concatenate([c[2][o].mean(0) for o in TEST]))
    return np.concatenate(out)


def fidelity(get_a, ps_a, get_b, ps_b):
    rows = []
    for pa, pb in product(ps_a, ps_b):
        if get_a is get_b and pa >= pb:
            continue
        a, b = neuron_means(get_a, pa), neuron_means(get_b, pb)
        if a is None or b is None:
            continue
        rows.append({"pa": pa, "pb": pb, "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "mean_abs_diff": float(np.mean(np.abs(a - b))), "mean_a": float(a.mean()), "mean_b": float(b.mean())})
    return rows


def agg(res):
    """mean/sd over Poisson seeds of single-seed R@1 (averaged over model seeds), ensemble, SL4."""
    if not res:
        return None
    r1 = [np.mean([v["seeds"][s]["R1"] for s in SEEDS]) for v in res.values()]
    r1_each = {s: [v["seeds"][s]["R1"] for v in res.values()] for s in SEEDS}
    r5 = [np.mean([v["seeds"][s]["R5"] for s in SEEDS]) for v in res.values()]
    sl4 = [np.mean([v["seeds"][s]["seq"][4] for s in SEEDS]) for v in res.values()]
    ens = [v["ens_R1"] for v in res.values()]
    ens4 = [v["ens_seq"][4] for v in res.values()]
    f = lambda x: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "n": len(x),
                   "values": [float(v) for v in x]}
    return {"R1": f(r1), "R5": f(r5), "SL4": f(sl4), "ens_R1": f(ens), "ens_SL4": f(ens4),
            "R1_by_model_seed": {s: f(v) for s, v in r1_each.items()}}


def main():
    b_ps = sorted({0} | {int(x.split("_P")[-1]) for x in glob.glob(os.path.join(ROOT, "e1", "S0_P*"))})
    get_b = brian2_counts
    summary = {"brian2": {"poisson_seeds": b_ps, "per_p": score_source(get_b, b_ps)}}
    summary["brian2"]["agg"] = agg(summary["brian2"]["per_p"])
    cfgs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "e3", "counts", "*")))
    summary["configs"] = {}
    for cfg in cfgs:
        ps = sorted(int(os.path.basename(f)[1:-4]) for f in glob.glob(os.path.join(ROOT, "e3", "counts", cfg, "P*.npz")))
        get = (lambda c: (lambda s, p: engine_counts(c, s, p)))(cfg)
        res = score_source(get, ps)
        summary["configs"][cfg] = {"poisson_seeds": ps, "per_p": res, "agg": agg(res),
                                   "fidelity_vs_brian2": fidelity(get, ps, get_b, b_ps) if cfg in ("a_float", "l_hw") else None}
        print(cfg, summary["configs"][cfg]["agg"] and {k: round(summary["configs"][cfg]["agg"][k]["mean"], 4) for k in ["R1", "ens_R1", "SL4"]}, flush=True)
    summary["fidelity_brian2_vs_brian2"] = fidelity(get_b, b_ps, get_b, b_ps)
    if "a_float" in cfgs:
        g = lambda s, p: engine_counts("a_float", s, p)
        summary["fidelity_float_vs_float"] = fidelity(g, summary["configs"]["a_float"]["poisson_seeds"], g,
                                                      summary["configs"]["a_float"]["poisson_seeds"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("brian2", {k: round(summary["brian2"]["agg"][k]["mean"], 4) for k in ["R1", "ens_R1", "SL4"]} if summary["brian2"]["agg"] else None)


if __name__ == "__main__":
    main()
