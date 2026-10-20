"""J2/J5 (Jetson Orin Nano): CPU latency, instruction count and power of the C golden model.

Same workload as scripts/e5_bench.py on the DGX: module E1 seed 0 offset 50 in the hardware
config (= golden_vectors/module_0.bin, delay seed 351), first test query (e1/S0/images.npz
test_x[0] = golden_vectors/queries.u8[0]), default stream 12345, delay mode 3.

  1. single core (pinned), dt = 0.1 and 0.5 ms: warm-up, then `reps` timed runs, with the
     power/clock sampler (scripts/orin_power.py) running on another core;
  2. instructions/cycles per run with tools/perfcount (perf_event_open), reps=3 minus reps=2;
  3. snn_golden_verify (80 different runs x reps) for a second median;
  4. parallel: one process per core on `par_cores`, each running the module loop (one 4-module
     query = 4 module runs in parallel on 4 cores);
  5. power: idle windows interleaved with ~60 s single-core and parallel load windows.
Writes results/orin_cpu.json; raw power CSVs go to logs/orin/power/.
Usage: python scripts/orin_cpu_bench.py [core] [reps]   (stdlib only, no numpy needed)
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GC = os.path.join(ROOT, "golden_c")
GV = os.path.join(ROOT, "golden_vectors")
MOD = os.path.join(GV, "module_0.bin")
IMG = os.path.join(ROOT, "cache", "query0.u8")
PLOG = os.path.join(ROOT, "logs", "orin", "power")
SAMPLER_CORE = 0
POWER = [sys.executable, os.path.join(ROOT, "scripts", "orin_power.py")]
os.makedirs(PLOG, exist_ok=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def temp_c():
    return int(open("/sys/devices/virtual/thermal/thermal_zone0/temp").read()) / 1000.0


def cool_down(limit=60.0, max_wait=300):
    t0 = time.time()
    while temp_c() > limit and time.time() - t0 < max_wait:
        time.sleep(5)
    return temp_c()


def start_sampler(name, seconds):
    out = os.path.join(PLOG, f"{name}.csv")
    p = subprocess.Popen(["taskset", "-c", str(SAMPLER_CORE)] + POWER + ["sample", out, str(seconds)])
    return p, out


def power_summary(csv, t0=None, t1=None):
    args = POWER + ["summary", csv] + ([str(t0), str(t1)] if t0 is not None else [])
    return json.loads(run(args).stdout)


def parse_bench(stdout):
    times = [float(x) for x in re.findall(r"time_s ([0-9.]+)", stdout)]
    sp = re.findall(r"e_spikes (\d+) in_spikes (\d+) i_spikes (\d+)", stdout)
    return times, sp


def bench_single(core, reps, dt_us):
    run(["taskset", "-c", str(core), os.path.join(GC, "snn_golden_bench"), MOD, IMG, "10", "3", str(dt_us)])  # warm-up
    temp0 = cool_down()
    smp, csv = start_sampler(f"j2_single_dt{dt_us}", 600)
    time.sleep(1.0)
    t0 = time.time()
    r = run(["taskset", "-c", str(core), os.path.join(GC, "snn_golden_bench"), MOD, IMG, str(reps), "3", str(dt_us)])
    t1 = time.time()
    smp.terminate(); smp.wait()
    times, sp = parse_bench(r.stdout)
    ps = power_summary(csv, t0, t1)
    return {"core": core, "median_s": statistics.median(times), "min_s": min(times), "max_s": max(times),
            "reps": len(times), "e_spikes": int(sp[0][0]), "in_spikes": int(sp[0][1]), "i_spikes": int(sp[0][2]),
            "all_spike_counts_identical": len(set(sp)) == 1,
            "cpu_freq_khz_during": ps[f"f_cpu{core}"], "temp_cpu_c_start": temp0,
            "temp_cpu_mC_during": ps.get("T_cpu-thermal"), "power_csv": os.path.relpath(csv, ROOT)}


def insns(core, dt_us):
    res = {}
    for reps in (2, 3):
        r = run([os.path.join(ROOT, "tools", "perfcount"), "taskset", "-c", str(core),
                 os.path.join(GC, "snn_golden_bench"), MOD, IMG, str(reps), "3", str(dt_us)])
        res[reps] = {k: int(v) for k, v in re.findall(r"(instructions|cycles) (\d+)", r.stderr)}
    d_i = res[3]["instructions"] - res[2]["instructions"]
    d_c = res[3]["cycles"] - res[2]["cycles"]
    return {"aarch64_instructions_per_run": d_i, "cycles_per_run": d_c, "ipc": d_i / d_c if d_c else None,
            "raw": res, "method": "tools/perfcount (perf_event_open, user-space only), reps3 - reps2"}


def verify(core, reps=3):
    r = run(["taskset", "-c", str(core), os.path.join(GC, "snn_golden_verify"), GV, str(reps)])
    m = re.search(r"(ALL PASS|FAIL); module runs timed: (\d+); median_s ([0-9.]+) min_s ([0-9.]+) max_s ([0-9.]+)", r.stdout)
    return {"result": m.group(1), "runs_timed": int(m.group(2)), "median_s": float(m.group(3)),
            "min_s": float(m.group(4)), "max_s": float(m.group(5)), "stdout": r.stdout.strip().splitlines()}


def loaded_window(name, cores, seconds_target, per_run_s, dt_us=100):
    """Run the module loop on each core for about seconds_target; sample power throughout."""
    reps = max(20, int(seconds_target / per_run_s))
    cool_down()
    smp, csv = start_sampler(name, seconds_target * 3 + 30)
    time.sleep(1.0)
    t0 = time.time()
    procs = [subprocess.Popen(["taskset", "-c", str(c), os.path.join(GC, "snn_golden_bench"), MOD, IMG, str(reps), "3", str(dt_us)],
                              stdout=subprocess.PIPE, text=True) for c in cores]
    outs = [p.communicate()[0] for p in procs]
    t1 = time.time()
    smp.terminate(); smp.wait()
    per_core = []
    for c, o in zip(cores, outs):
        times, sp = parse_bench(o)
        per_core.append({"core": c, "median_s": statistics.median(times), "reps": len(times),
                         "e_spikes": int(sp[0][0]), "all_spike_counts_identical": len(set(sp)) == 1})
    total_runs = sum(pc["reps"] for pc in per_core)
    return {"cores": cores, "wall_s": t1 - t0, "module_runs": total_runs, "module_runs_per_s": total_runs / (t1 - t0),
            "per_core": per_core, "power": power_summary(csv, t0, t1), "power_csv": os.path.relpath(csv, ROOT)}


def idle_window(name, seconds=60):
    cool_down()
    smp, csv = start_sampler(name, seconds)
    smp.wait()
    return {"seconds": seconds, "power": power_summary(csv), "power_csv": os.path.relpath(csv, ROOT)}


def main(core=3, reps=25):
    out = {"module": "E1 seed 0, offset 50 (golden_vectors/module_0.bin, delay seed 351)",
           "query": "first test query (seed 0), stream 12345", "uptime_before": run(["uptime"]).stdout.strip()}
    for name, dt_us in [("l_hw", 100), ("l_hw_dt05", 500)]:
        r = {"single": bench_single(core, reps, dt_us)}
        r.update(insns(core, dt_us))
        out[name] = r
        print(name, json.dumps(r["single"])[:400], r["aarch64_instructions_per_run"], flush=True)
    out["verify"] = verify(core)
    print("verify", out["verify"]["result"], out["verify"]["median_s"], flush=True)
    per = out["l_hw"]["single"]["median_s"]
    par = [2, 3, 4, 5]
    # interleave: idle, single, idle, parallel-4, idle
    out["power"] = {"idle_1": idle_window("j5_idle_1"),
                    "single_core": loaded_window("j5_single", [core], 60, per),
                    "idle_2": idle_window("j5_idle_2"),
                    "parallel_4": loaded_window("j5_par4", par, 60, per),
                    "idle_3": idle_window("j5_idle_3")}
    out["uptime_after"] = run(["uptime"]).stdout.strip()
    with open(os.path.join(ROOT, "results", "orin_cpu.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/orin_cpu.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 3, int(a[1]) if len(a) > 1 else 25)
