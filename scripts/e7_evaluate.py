"""E7 scoring of the released full-Nordland instance with VPRSNN's own evaluation logic.

Per engine config: hyperactive threshold calibrated on calibration modules/queries
(places 0..599; best R@1, ties -> smallest), then R@1/R@5 and SeqSLAM (Lseq 2,4,10) on the
2700 test queries. Place order is unshuffled (identity indices). Reference responses for the
hyperactive mask: 'brian2rec' = released Brian2 resultPopVecs3300.npy (spring);
'enginerec' = the engine's own spring responses (if e7/counts/<cfg>/record_c*.npz exist).
Writes results/e7_<cfg>.json.  Usage: python scripts/e7_evaluate.py CONFIG
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from snnsim import evaluate as EV  # noqa: E402
from e7_run import CAL_MODS, TEST_MODS, NCAL, NPL, REL, load_rel  # noqa: E402


def load_counts(cfg, session):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "e7", "counts", cfg, f"{session}_c*.npz"))):
        z = np.load(f)
        for m, c in zip(z["modules"], z["counts"]):
            out[int(m)] = c.astype(np.int64)
    return out


def main(cfg):
    base = cfg.split("_P")[0] if "_P" in cfg else cfg
    cal, test = load_counts(cfg, "cal"), load_counts(cfg, "test")
    missing = [m for m in CAL_MODS if m not in cal] + [m for m in TEST_MODS if m not in test]
    if missing:
        print("missing modules", missing[:10], len(missing)); return
    A = {m: load_rel(m)[2] for m in CAL_MODS + TEST_MODS}
    recs = {"brian2rec": {m: np.load(f"{REL}/outputs/outputs_ne400_L25_offset{m}/resultPopVecs3300.npy") for m in CAL_MODS + TEST_MODS}}
    er = load_counts(cfg, "record")
    if all(m in er for m in CAL_MODS + TEST_MODS):
        recs["enginerec"] = er
    ident = np.arange(NPL)
    res = {"config": cfg, "n_cal_modules": len(CAL_MODS), "n_test_modules": len(TEST_MODS),
           "n_cal_queries": NCAL, "n_test_queries": NPL - NCAL}
    for name, R in recs.items():
        th, table = EV.calibrate_threshold([R[m] for m in CAL_MODS], [cal[m] for m in CAL_MODS],
                                           [A[m] for m in CAL_MODS], ident, NCAL)
        r, sr = EV.evaluate_split([R[m] for m in TEST_MODS], [test[m] for m in TEST_MODS],
                                  [A[m] for m in TEST_MODS], ident, NCAL, "test", th)
        r0, _ = EV.evaluate_split([R[m] for m in TEST_MODS], [test[m] for m in TEST_MODS],
                                  [A[m] for m in TEST_MODS], ident, NCAL, "test", 0)
        seq = {L: EV.seqslam_recall(sr, L)[1] for L in (2, 4, 10)}
        res[name] = {"threshold": th, "cal_R1_by_threshold": table, "R1": r[1], "R5": r[5],
                     "R1_threshold0": r0[1], "seq_R1": seq}
        print(cfg, name, json.dumps(res[name])[:300])
    with open(os.path.join(ROOT, "results", f"e7_{cfg}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
