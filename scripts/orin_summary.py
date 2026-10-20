"""Collect the Jetson Orin Nano results into results/orin_summary.json (keyed, labelled, sourced,
like reports/results.json) and print the tables used in reports/orin_baseline_report.md.

Energy (DERIVED): for a load window with mean power P_load and throughput R module runs/s,
  above idle  E_run = (P_load - P_idle) / R
  total       E_run = P_load / R
P_idle = mean of the idle windows taken in the same script run. Per 4-module query = 4 x E_run,
except where the 4 modules run concurrently (stated per entry).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
OUT = {}


def put(key, value, label, source, note=None, dgx=None):
    OUT[key] = {"value": value, "label": label, "source": source}
    if note:
        OUT[key]["note"] = note
    if dgx is not None:
        OUT[key]["dgx"] = dgx


def pw(win, rail="P_VDD_IN"):
    return win["power"][rail]["mean"] / 1000.0


def energy(prefix, load_w, idle_w, rate, src, cg=None, cg_idle=None, query_factor=4, query_note=None):
    put(f"{prefix}.power_W.VDD_IN", load_w, "MEASURED", src)
    put(f"{prefix}.power_W.idle_VDD_IN", idle_w, "MEASURED", src)
    if cg is not None:
        put(f"{prefix}.power_W.VDD_CPU_GPU_CV", cg, "MEASURED", src)
        put(f"{prefix}.power_W.idle_VDD_CPU_GPU_CV", cg_idle, "MEASURED", src)
    put(f"{prefix}.module_runs_per_s", rate, "MEASURED", src)
    put(f"{prefix}.energy_mJ_per_run.above_idle", (load_w - idle_w) / rate * 1e3, "DERIVED", src, "(P_load - P_idle) / rate, VDD_IN")
    put(f"{prefix}.energy_mJ_per_run.total", load_w / rate * 1e3, "DERIVED", src, "P_load / rate, VDD_IN")
    put(f"{prefix}.energy_mJ_per_query.above_idle", query_factor * (load_w - idle_w) / rate * 1e3, "DERIVED", src,
        query_note or f"{query_factor} x per-run above-idle energy")
    put(f"{prefix}.energy_mJ_per_query.total", query_factor * load_w / rate * 1e3, "DERIVED", src,
        query_note or f"{query_factor} x per-run total energy")


def main():
    dgx_bench, dgx_b2, dgx_gpu = R("e5_bench.json"), R("e1_brian2_clean_timing.json"), R("e5_gpu.json")
    # ---- C golden model, CPU
    c = R("orin_cpu.json"); s = "results/orin_cpu.json"
    for cfg in ("l_hw", "l_hw_dt05"):
        x = c[cfg]["single"]
        put(f"C.{cfg}.a78ae.median_s", x["median_s"], "MEASURED", s,
            f"core {x['core']}, {x['reps']} reps, CPU {x['cpu_freq_khz_during']['p50'] / 1e3:.0f} MHz (p50), MAXN_SUPER",
            {"x925": dgx_bench[cfg]["x925"]["median_s"], "a725": dgx_bench[cfg]["a725"]["median_s"]})
        put(f"C.{cfg}.spikes", [x["e_spikes"], x["in_spikes"], x["i_spikes"]], "MEASURED", s, "E, input, I",
            [dgx_bench[cfg]["x925"]["e_spikes"], dgx_bench[cfg]["x925"]["in_spikes"], dgx_bench[cfg]["x925"]["i_spikes"]])
        put(f"C.{cfg}.aarch64_instructions_per_run", c[cfg]["aarch64_instructions_per_run"], "MEASURED", s,
            "perf_event_open user-space, reps3 - reps2, gcc 11.4 -O3", dgx_bench[cfg]["aarch64_instructions_per_run"])
        put(f"C.{cfg}.ipc", c[cfg]["ipc"], "MEASURED", s)
    put("C.verify.median_s", c["verify"]["median_s"], "MEASURED", s, f"{c['verify']['runs_timed']} runs (4 modules x 20 queries x 3), {c['verify']['result']}")
    p = c["power"]; idle = sum(pw(p[k]) for k in ("idle_1", "idle_2", "idle_3")) / 3
    idle_cg = sum(pw(p[k], "P_VDD_CPU_GPU_CV") for k in ("idle_1", "idle_2", "idle_3")) / 3
    sc = p["single_core"]
    energy("C.single_core", pw(sc), idle, sc["module_runs_per_s"], s, pw(sc, "P_VDD_CPU_GPU_CV"), idle_cg)
    p4 = p["parallel_4"]
    energy("C.parallel_4cores", pw(p4), idle, p4["module_runs_per_s"], s, pw(p4, "P_VDD_CPU_GPU_CV"), idle_cg,
           query_note="4 x per-run energy; the 4 modules of a query run concurrently on cores 2-5")
    put("C.parallel_4cores.query_latency_s", max(x["median_s"] for x in p4["per_core"]), "MEASURED", s,
        "4 modules of one query on 4 cores in parallel = slowest per-core median module run")
    # ---- Brian2
    b = R("orin_brian2_timing.json"); s = "results/orin_brian2_timing.json"
    put("Brian2.a78ae.median_s_per_image", b["median_s_per_image"], "MEASURED", s, "VPRSNN test mode, E1 S0 offset 50, core 3",
        {"x925": dgx_b2["x925"]["median_s_per_image"], "a725": dgx_b2["a725"]["median_s_per_image"]})
    put("Brian2.a78ae.present_350ms_median_s", b["present_350ms_median_s"], "MEASURED", s, None,
        {"x925": dgx_b2["x925"]["present_350ms_median_s"], "a725": dgx_b2["a725"]["present_350ms_median_s"]})
    put("Brian2.a78ae.rest_150ms_median_s", b["rest_150ms_median_s"], "MEASURED", s, None,
        {"x925": dgx_b2["x925"]["rest_150ms_median_s"], "a725": dgx_b2["a725"]["rest_150ms_median_s"]})
    put("Brian2.output_identical_to_dgx", b["result_vs_dgx"], "MEASURED", s, "resultPopVecs100_test_E30.npy, offset 50")
    # ---- CUDA port
    g = R("orin_cuda.json"); s = "results/orin_cuda.json"
    for dt in (100, 500):
        bb = g[f"bench_dt{dt}"]
        put(f"CUDA.dt{dt}.single_run_kernel_s", bb["single_run"]["kernel_median_s"], "MEASURED", s, "module_0, query 0, stream 12345; batch 1")
        put(f"CUDA.dt{dt}.query_4_modules_kernel_s", bb["query_4_modules"]["kernel_median_s"], "MEASURED", s, "4 modules of query 0 in one launch")
        put(f"CUDA.dt{dt}.query_4_modules_e2e_s", bb["query_4_modules"]["e2e_median_s"], "MEASURED", s, "incl. run-list upload and count copy-back")
        put(f"CUDA.dt{dt}.best_module_runs_per_s", max(x["module_runs_per_s"] for x in bb["batches"]), "MEASURED", s)
        put(f"CUDA.dt{dt}.bit_exact", all(x["mismatched_runs"] == 0 for x in bb["batches"]) and bb["single_run"]["bit_exact_vs_c"], "MEASURED", s,
            "every run of every batch compared with the C model")
    w = g["power"]
    for name, idle_keys in (("loop_batch4", ("idle_1", "idle_after_loop_batch4")),
                            ("loop_batch96", ("idle_after_loop_batch4", "idle_after_loop_batch96"))):
        iw = sum(pw(w[k]) for k in idle_keys) / 2
        icg = sum(pw(w[k], "P_VDD_CPU_GPU_CV") for k in idle_keys) / 2
        energy(f"CUDA.{name}", pw(w[name]), iw, w[name]["module_runs_per_s"], s, pw(w[name], "P_VDD_CPU_GPU_CV"), icg,
               query_note="4 x per-run energy" + ("; one launch = one query (4 modules concurrently)" if name == "loop_batch4" else ""))
    # ---- torch tooling engine
    t = R("orin_gpu.json"); s = "results/orin_gpu.json"
    it = pw(t["idle_after"])
    for r in t["runs"]:
        energy(f"torch.{r['dtype']}.batch{r['batch']}", r["power"]["P_VDD_IN"]["mean"] / 1000, it, r["module_runs_per_s"], s,
               r["power"]["P_VDD_CPU_GPU_CV"]["mean"] / 1000, pw(t["idle_after"], "P_VDD_CPU_GPU_CV"))
        OUT[f"torch.{r['dtype']}.batch{r['batch']}.module_runs_per_s"]["note"] = f"mean E spikes {r['mean_e_spikes']:.2f}"
    put("torch.float32.best_module_runs_per_s", t["best_module_runs_per_s"], "MEASURED", s, t["label"], dgx_gpu["best_module_runs_per_s"])
    # ---- J6
    for f in ("orin_e4_l_hw.json", "orin_e3_summary.json"):
        if os.path.exists(os.path.join(ROOT, "results", f)):
            put(f"J6.{f}", "see file", "MEASURED", f"results/{f}")
    with open(os.path.join(ROOT, "results", "orin_summary.json"), "w") as fo:
        json.dump(OUT, fo, indent=1)
    for k, v in OUT.items():
        val = v["value"]
        val = f"{val:.6g}" if isinstance(val, float) else val
        print(f"{k:58s} {v['label']:8s} {val}" + (f"   DGX {v['dgx']}" if "dgx" in v else ""))


if __name__ == "__main__":
    main()
