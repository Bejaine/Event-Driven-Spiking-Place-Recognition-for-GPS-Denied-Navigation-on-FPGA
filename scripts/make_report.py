"""Collect every reported number into reports/results.json (with labels and source paths)
and draw reports/figures/*. Run after the experiment scripts."""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "reports", "figures")
sys.path.insert(0, ROOT)

out = {}


def put(key, value, label, source, note=None):
    d = {"value": value, "label": label, "source": os.path.relpath(source, ROOT) if os.path.isabs(source) else source}
    if note:
        d["note"] = note
    out[key] = d


def jl(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def e1():
    b = {P: jl(f"e1_brian2_P{P}.json") for P in range(4)}
    b = {k: v for k, v in b.items() if v}
    if 0 not in b:
        return
    src = os.path.join(RES, "e1_brian2_P0.json")
    for s, v in b[0]["seeds"].items():
        put(f"E1.brian2.P0.seed{s}.R1", v["test"]["1"], "MEASURED", src)
        put(f"E1.brian2.P0.seed{s}.R5", v["test"]["5"], "MEASURED", src)
        put(f"E1.brian2.P0.seed{s}.threshold", v["threshold"], "MEASURED", src)
        for L in ("2", "4", "10"):
            put(f"E1.brian2.P0.seed{s}.SL{L}.R1", v["seq"][L]["1"], "MEASURED", src)
    put("E1.brian2.P0.ens3.R1", b[0]["ensemble"]["SL1"]["1"], "MEASURED", src)
    for L in ("2", "4", "10"):
        put(f"E1.brian2.P0.ens3.SL{L}.R1", b[0]["ensemble"]["seq"][L]["1"], "MEASURED", src)
    # noise band over Poisson seeds (repo-script results)
    ps = sorted(b)
    r1 = {s: [b[p]["seeds"][s]["test"]["1"] for p in ps] for s in b[0]["seeds"]}
    mean_r1 = [np.mean([b[p]["seeds"][s]["test"]["1"] for s in b[0]["seeds"]]) for p in ps]
    ens = [b[p]["ensemble"]["SL1"]["1"] for p in ps]
    srcs = ", ".join(f"results/e1_brian2_P{p}.json" for p in ps)
    put("E1.brian2.noise.poisson_seeds", ps, "MEASURED", srcs)
    put("E1.brian2.noise.meanR1_over_seeds.per_P", [float(x) for x in mean_r1], "MEASURED", srcs)
    put("E1.brian2.noise.meanR1.mean_sd", [float(np.mean(mean_r1)), float(np.std(mean_r1, ddof=1))], "DERIVED", srcs, "mean, sd over Poisson seeds of the 3-seed mean R@1")
    for s, v in r1.items():
        put(f"E1.brian2.noise.seed{s}.R1.min_max", [min(v), max(v)], "MEASURED", srcs)
    put("E1.brian2.noise.ens3.R1.per_P", ens, "MEASURED", srcs)
    # timing
    tr, inf = {"x925": [], "a725": []}, {"x925": [], "a725": []}
    x925 = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}
    for f in glob.glob(os.path.join(ROOT, "e1", "S*", "timing", "O*_*.json")):
        if "_P" in f.split(os.sep)[-3]:
            continue
        d = json.load(open(f))
        core = d["cpu"][0] if isinstance(d["cpu"], list) else None
        kind = "x925" if core in x925 else "a725"
        for c in d["calls"]:
            runs = np.array(c["net_runs"]) if c["net_runs"] else np.zeros((0, 2))
            if c["process_mode"] == "train":
                pres = runs[np.isclose(runs[:, 0], 0.35), 1].sum() + runs[np.isclose(runs[:, 0], 0.15), 1].sum()
                tr[kind].append({"wall_s": c["wall_s"], "sim_runs_s": float(pres), "n_img": int(np.isclose(runs[:, 0], 0.35).sum())})
            else:
                p = runs[np.isclose(runs[:, 0], 0.35), 1]; r = runs[np.isclose(runs[:, 0], 0.15), 1]
                n = min(len(p), len(r))
                if n:
                    inf[kind].extend((p[:n] + r[:n]).tolist())
    for k in tr:
        if tr[k]:
            w = [x["wall_s"] for x in tr[k]]
            put(f"E1.brian2.train_wall_s_per_module.{k}", {"median": float(np.median(w)), "min": float(min(w)), "max": float(max(w)), "n": len(w),
                                                            "images_per_module": tr[k][0]["n_img"]}, "MEASURED", "e1/S*/timing/O*_train.json",
                "18 modules trained concurrently, one process pinned per core (load ~18 of 20 cores)")
        if inf[k]:
            put(f"E1.brian2.infer_s_per_image_per_module.{k}", {"median": float(np.median(inf[k])), "p10": float(np.percentile(inf[k], 10)),
                                                                 "p90": float(np.percentile(inf[k], 90)), "n": len(inf[k])}, "MEASURED",
                "e1/S*/timing/O*_*.json", "Brian2 net.run(350 ms) + net.run(150 ms rest) per image, test-mode network, excludes network build/codegen")


def e3():
    s = jl("e3_summary.json")
    if not s:
        return
    src = os.path.join(RES, "e3_summary.json")
    b = s["brian2"]["agg"]
    put("E3.brian2.agg", b, "MEASURED", src, "mean/sd over Poisson seeds; R1 = mean over 3 model seeds")
    for cfg, v in s["configs"].items():
        if v["agg"]:
            a = v["agg"]
            d = a["R1"]["mean"] - b["R1"]["mean"]
            band = 2 * np.sqrt(b["R1"]["sd"] ** 2 + a["R1"]["sd"] ** 2)
            ok = abs(d) <= 0.02 and abs(d) <= max(band, 0.01)
            put(f"E3.{cfg}.agg", a, "MEASURED", src)
            put(f"E3.{cfg}.dR1_vs_brian2", float(d), "DERIVED", src, "mean R1(cfg) - mean R1(Brian2)")
            put(f"E3.{cfg}.pass", bool(ok), "DERIVED", src, "pass = |dR1| <= 2 pp AND |dR1| <= max(2*sqrt(sd_b^2+sd_c^2), 1 pp)")
        if v.get("fidelity_vs_brian2"):
            put(f"E2.fidelity.{cfg}_vs_brian2", v["fidelity_vs_brian2"], "MEASURED", src)
    put("E2.fidelity.brian2_vs_brian2", s.get("fidelity_brian2_vs_brian2"), "MEASURED", src)
    put("E2.fidelity.float_vs_float", s.get("fidelity_float_vs_float"), "MEASURED", src)


def simple(prefix, name, label="MEASURED"):
    d = jl(name)
    if d:
        put(prefix, d, label, os.path.join(RES, name))


def figures():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    s = jl("e3_summary.json")
    if s:
        names = [k for k in ["a_float", "b_w8", "c_w4", "c_w4max", "d_hash", "e_hash1ms", "e_hash1ms_binned", "f_nodelay",
                             "g_dt02", "g_dt05", "h_hard", "i_shared", "j_lfsr", "k_fixed", "l_hw", "l_hw_w4", "l_hw_w4max", "l_hw_dt05"]
                 if k in s["configs"] and s["configs"][k]["agg"]]
        m = [s["configs"][k]["agg"]["R1"]["mean"] for k in names]
        e = [s["configs"][k]["agg"]["R1"]["sd"] for k in names]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(range(len(names)), m, yerr=e, color=["#4c72b0" if not k.startswith("l_") else "#dd8452" for k in names], capsize=3)
        b = s["brian2"]["agg"]["R1"]
        ax.axhspan(b["mean"] - 0.02, b["mean"] + 0.02, color="green", alpha=0.15, label="Brian2 mean ± 2 pp")
        ax.axhline(b["mean"], color="green", lw=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylim(min(m) - 0.1, 1.0); ax.set_ylabel("R@1 (mean over 3 model seeds; ±sd over Poisson seeds)")
        ax.set_title("E3 ablation, Nordland NRD_SFS, 100 test places (MEASURED)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "ablation_r1.png"), dpi=130); plt.close(fig)
    for cfg in ["l_hw", "a_float"]:
        p = os.path.join(RES, f"e4_{cfg}_traces.npz")
        if os.path.exists(p):
            ev = np.load(p)["events"].ravel()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ev, bins=np.arange(ev.max() + 2) - 0.5, color="#4c72b0")
            ax.set_yscale("log"); ax.set_xlabel("input events per 0.1 ms step"); ax.set_ylabel("steps")
            ax.set_title(f"E4 events per step ({cfg}, MEASURED)")
            fig.tight_layout(); fig.savefig(os.path.join(FIG, f"events_per_step_{cfg}.png"), dpi=130); plt.close(fig)
        p = os.path.join(RES, f"e4_{cfg}_state.npz")
        if os.path.exists(p):
            z = np.load(p)
            fig, axs = plt.subplots(1, 5, figsize=(16, 3.2))
            for ax, k in zip(axs, ["v_e", "ge_e", "gi_e", "v_i", "ge_i"]):
                x = z[k] * (z["v_scale"] if k.startswith("v") else z["g_scale"])
                ax.hist(x, bins=100, color="#55a868"); ax.set_yscale("log")
                ax.set_title(k + (" [mV]" if k.startswith("v") else ""))
            fig.suptitle(f"E4 sampled state values ({cfg}, MEASURED)")
            fig.tight_layout(); fig.savefig(os.path.join(FIG, f"state_ranges_{cfg}.png"), dpi=130); plt.close(fig)


def main():
    e1()
    e3()
    simple("E4.l_hw", "e4_l_hw.json")
    simple("E4.a_float", "e4_a_float.json")
    simple("E4.l_hw_dt05", "e4_l_hw_dt05.json")
    simple("E5", "e5_bench.json")
    simple("E5.gpu", "e5_gpu.json")
    simple("E6", "e6_cyclemodel.json", "DERIVED")
    for f in sorted(glob.glob(os.path.join(RES, "e7_*.json"))):
        simple("E7." + os.path.basename(f)[3:-5], os.path.basename(f))
    simple("E8", "e8_training_projection.json", "DERIVED")
    simple("E1.brian2.clean_timing", "e1_brian2_clean_timing.json")
    put("paper.TableI.ModularSNN.NordlandFS", {"SL1": 0.42, "SL2": 0.55, "SL4": 0.67, "SL10": 0.78},
        "EXTERNAL", "https://arxiv.org/html/2311.13186 (Table I)")
    put("paper.TableI.Ens3.NordlandFS", {"SL1": 0.50, "SL2": 0.62, "SL4": 0.72, "SL10": 0.80},
        "EXTERNAL", "https://arxiv.org/html/2311.13186 (Table I)")
    with open(os.path.join(ROOT, "reports", "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    figures()
    print("keys:", len(out))


if __name__ == "__main__":
    main()
