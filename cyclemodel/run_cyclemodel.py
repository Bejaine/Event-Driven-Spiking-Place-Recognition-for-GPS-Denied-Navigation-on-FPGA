"""E6: run the cycle model on the measured E4 event traces. All outputs are DERIVED.

Inputs : results/e4_l_hw_traces.npz (and e4_l_hw_dt05_traces.npz if present),
         results/e5_bench.json (MEASURED C baseline, U54 ESTIMATE)
Output : results/e6_cyclemodel.json, reports/figures/cycle_model.png
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cyclemodel as CM  # noqa: E402

LANES = [50, 100, 200]
FREQS = [100e6, 125e6, 150e6]
BWS = [0.5e9, 1.0e9, 2.0e9]
MULTS_PER_LANE = 2          # assumption: 2 hardware multipliers per lane (sensitivity: 1 and 5 below)
ENGINES_RESIDENT = 4


def main():
    res = {"assumptions": {"mults_per_lane": MULTS_PER_LANE, "MUL_E": CM.MUL_E, "MUL_I": CM.MUL_I,
                           "overhead_cycles_per_step": CM.OVERHEAD, "spike_gen": "16 xorshift32 generators x 4 draws/cycle",
                           "module_run": "hard reset + 350 ms presentation (no rest)"}}
    e5 = json.load(open(os.path.join(ROOT, "results", "e5_bench.json")))
    total_bytes, wbytes, meta = CM.module_bytes(8)
    total_bytes4, wbytes4, _ = CM.module_bytes(4)
    res["module_bytes_8bit"] = {"total": total_bytes, "weights": wbytes, "meta": meta}
    res["module_bytes_4bit"] = {"total": total_bytes4, "weights": wbytes4}
    for tag, cfg_key in [("dt0.1ms", "l_hw"), ("dt0.5ms", "l_hw_dt05")]:
        p = os.path.join(ROOT, "results", f"e4_{cfg_key}_traces.npz")
        if not os.path.exists(p):
            continue
        ev = np.load(p)["events"]
        c_x925 = e5.get(cfg_key, {}).get("x925", {}).get("median_s")
        u54 = e5.get(cfg_key, {}).get("u54_estimate", {})
        rows = []
        for L in LANES:
            cyc = np.array([CM.cycles_per_run(e, L, MULTS_PER_LANE)[0] for e in ev])
            info = CM.cycles_per_run(ev[0], L, MULTS_PER_LANE)[1]
            for f in FREQS:
                t_run = cyc / f
                row = {"L": L, "f_MHz": f / 1e6, "cycles_per_run_mean": float(cyc.mean()),
                       "cycles_per_run_max": float(cyc.max()), "t_run_mean_ms": float(t_run.mean() * 1e3),
                       "t_run_max_ms": float(t_run.max() * 1e3),
                       "resident_t_query_ms": float(t_run.mean() * 1e3),
                       "resident_queries_per_s": float(1 / t_run.mean()),
                       "upd_cycles_per_step": info["upd_cycles"], "gen_cycles_per_step": info["gen_cycles"],
                       "frac_steps_accumulation_bound": float(np.mean([CM.cycles_per_run(e, L, MULTS_PER_LANE)[1]["frac_steps_acc_bound"] for e in ev[:200]])),
                       "speedup_vs_C_X925_single_core": (c_x925 / t_run.mean()) if c_x925 else None,
                       "speedup_vs_U54_est_ipc1": (u54.get("time_s_ipc_1.0") / t_run.mean()) if u54.get("time_s_ipc_1.0") else None,
                       "speedup_vs_U54_est_ipc0.5": (u54.get("time_s_ipc_0.5") / t_run.mean()) if u54.get("time_s_ipc_0.5") else None,
                       "dsp_multipliers_resident": ENGINES_RESIDENT * L * MULTS_PER_LANE}
                row["streaming"] = {}
                for bw in BWS:
                    s = CM.streaming(float(t_run.mean()), bw, total_bytes, 60, 2)
                    row["streaming"][f"{bw / 1e9:g}GBps"] = s
                rows.append(row)
        sens = {}
        for m in (1, 5):
            cyc = np.array([CM.cycles_per_run(e, 100, m)[0] for e in ev])
            sens[f"L100_mults{m}_cycles_mean"] = float(cyc.mean())
        res[tag] = {"n_traces": int(len(ev)), "steps_per_run": int(ev.shape[1]), "rows": rows,
                    "sensitivity_mults_per_lane": sens, "c_x925_median_s": c_x925, "u54": u54}
    lsr = {}
    for L in LANES:
        b8, d8, w8 = CM.lsram_blocks(L, 8)
        b4, d4, w4 = CM.lsram_blocks(L, 4)
        lsr[L] = {"8bit": {"depth": d8, "width": w8, "blocks_per_bank": b8, "blocks_4_banks": {k: 4 * v for k, v in b8.items()}},
                  "4bit": {"depth": d4, "width": w4, "blocks_per_bank": b4, "blocks_4_banks": {k: 4 * v for k, v in b4.items()}}}
    res["lsram"] = {"device_blocks_MPFS250T": 812, "by_lanes": lsr,
                    "ring_buffer_bits_per_engine": {"0.1ms_101slots": CM.ring_buffer_bits(101), "1ms_binned_12slots": CM.ring_buffer_bits(12)},
                    "ring_buffer_lsram_blocks_per_engine": {"0.1ms_101slots": math.ceil(CM.ring_buffer_bits(101) / 20480),
                                                            "1ms_binned_12slots": math.ceil(CM.ring_buffer_bits(12) / 20480)}}
    with open(os.path.join(ROOT, "results", "e6_cyclemodel.json"), "w") as f:
        json.dump(res, f, indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for tag, mk in [("dt0.1ms", "o-"), ("dt0.5ms", "s--")]:
            if tag not in res:
                continue
            for f in FREQS:
                r = [x for x in res[tag]["rows"] if x["f_MHz"] == f / 1e6]
                ax[0].plot([x["L"] for x in r], [x["t_run_mean_ms"] for x in r], mk, label=f"{tag} {f/1e6:.0f} MHz")
            r = [x for x in res[tag]["rows"] if x["f_MHz"] == 125]
            for bw in BWS:
                ax[1].plot([x["L"] for x in r], [x["streaming"][f"{bw/1e9:g}GBps"]["queries_per_s"] for x in r], mk,
                           label=f"{tag} {bw/1e9:g} GB/s")
        ax[0].set_xlabel("lanes L"); ax[0].set_ylabel("module run time [ms] (DERIVED)"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("lanes L"); ax[1].set_ylabel("queries/s, 500 places x 3 (125 MHz)"); ax[1].legend(fontsize=7)
        fig.suptitle("SpikeNav cycle model (driven by measured E4 event traces)")
        fig.tight_layout()
        os.makedirs(os.path.join(ROOT, "reports", "figures"), exist_ok=True)
        fig.savefig(os.path.join(ROOT, "reports", "figures", "cycle_model.png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot failed", e)
    print(json.dumps({k: v for k, v in res.items() if k != "lsram"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
