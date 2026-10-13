"""Check that snnsim/evaluate.py (in-memory) reproduces the repo scripts' on-disk results
(results/e1_brian2_P0.json) on the Brian2 E1 spike counts: thresholds, R@1/R@5, SeqSLAM, ensemble."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("TMPDIR", os.path.join(ROOT, "cache"))
import e3_evaluate as E3
ref = json.load(open(os.path.join(ROOT, "results", "e1_brian2_P0.json")))
mine = E3.score_source(E3.brian2_counts, [0])[0]
ok = True
for s in E3.SEEDS:
    r, m = ref["seeds"][str(s)], mine["seeds"][s]
    same = (r["threshold"] == m["threshold"] and abs(r["test"]["1"] - m["R1"]) < 1e-12 and abs(r["test"]["5"] - m["R5"]) < 1e-12
            and all(abs(r["seq"][str(L)]["1"] - m["seq"][L]) < 1e-12 for L in (2, 4, 10)))
    print(f"seed {s}: repo thr {r['threshold']} R1 {r['test']['1']} R5 {r['test']['5']} SL4 {r['seq']['4']['1']} | "
          f"in-memory thr {m['threshold']} R1 {m['R1']} R5 {m['R5']} SL4 {m['seq'][4]} -> {'MATCH' if same else 'DIFF'}")
    ok &= same
e_same = abs(ref["ensemble"]["SL1"]["1"] - mine["ens_R1"]) < 1e-12 and abs(ref["ensemble"]["seq"]["4"]["1"] - mine["ens_seq"][4]) < 1e-12
print(f"ensemble: repo R1 {ref['ensemble']['SL1']['1']} SL4 {ref['ensemble']['seq']['4']['1']} | in-memory {mine['ens_R1']} {mine['ens_seq'][4]} -> {'MATCH' if e_same else 'DIFF'}")
print("ALL MATCH" if ok and e_same else "MISMATCH")
