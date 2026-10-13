"""E5: CPU baselines of the C golden model (single core) + instruction counts + U54 estimate.

  * builds golden_c (gcc -O3, C99, no intrinsics)
  * module = E1 seed-0 test module (offset 50) in the proposed hardware configuration
    (8-bit weights, hashed 1 ms binned delays, hard reset, xorshift input), first test query
  * median of N timed module runs pinned to a Cortex-X925 core and to a Cortex-A725 core
  * instructions per module run: valgrind/callgrind Ir (perf is blocked on this machine),
    computed as Ir(reps=3) - Ir(reps=2) to remove setup cost
  * RV64 instructions: riscv64 cross build under qemu-riscv64 with the insn-count plugin,
    if the user-space toolchain in tools/ is available
Writes results/e5_bench.json. Usage: python scripts/e5_bench.py [reps]
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "golden_c")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from snnsim import fixed as FX  # noqa: E402
from snnsim.configs import CONFIGS  # noqa: E402
from snnsim.modules import load_seed  # noqa: E402
from e3_run import module_delay_seed  # noqa: E402
import pygolden  # noqa: E402

GC = os.path.join(ROOT, "golden_c")
X925, A725 = 5, 1
TOOLS = os.path.join(ROOT, "tools", "root")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def bench(core, reps, dmode, dt_us, mod, img):
    r = run(["taskset", "-c", str(core), os.path.join(GC, "snn_golden_bench"), mod, img, str(reps), str(dmode), str(dt_us)])
    times = [float(m) for m in re.findall(r"time_s ([0-9.]+)", r.stdout)]
    spikes = re.findall(r"e_spikes (\d+) in_spikes (\d+) i_spikes (\d+)", r.stdout)
    return {"median_s": statistics.median(times), "min_s": min(times), "max_s": max(times), "reps": reps,
            "e_spikes": int(spikes[0][0]), "in_spikes": int(spikes[0][1]), "i_spikes": int(spikes[0][2])}


def callgrind_ir(binary, args, prefix=None):
    vg = os.path.join(ROOT, "envs", "tools", "bin", "valgrind")
    out = os.path.join(ROOT, "cache", "cg.out")
    r = run([vg, "--tool=callgrind", f"--callgrind-out-file={out}", binary] + args)
    m = re.search(r"Collected : (\d+)", r.stderr)
    return int(m.group(1)) if m else None


def qemu_rv64_insns(mod, img, dmode, dt_us):
    """RV64 instruction count of the static riscv64 build under qemu-riscv64 (user mode).

    Ubuntu's qemu-user is built without TCG plugins, so instructions are counted from
    `-d in_asm,exec,nochain` logs: (instructions per translated block) x (executions),
    streamed through tools/qemu_count.py. Validated on a 1M-iteration 5-insn loop (5.14M).
    """
    gcc = os.path.join(TOOLS, "usr", "bin", "riscv64-linux-gnu-gcc-13")
    qemu = os.path.join(TOOLS, "usr", "bin", "qemu-riscv64")
    if not (os.path.exists(gcc) and os.path.exists(qemu)):
        return {"available": False}
    exe = os.path.join(GC, "snn_golden_bench_rv64")
    env = dict(os.environ, PATH=os.path.join(TOOLS, "usr", "bin") + ":" + os.environ["PATH"],
               LD_LIBRARY_PATH=os.path.join(TOOLS, "usr", "lib", "aarch64-linux-gnu"))
    sysroot = os.path.join(TOOLS, "usr", "riscv64-linux-gnu")
    flags = ["-O3", "-std=c99", "-static", "-march=rv64gc", "-mabi=lp64d"]
    c = run([gcc] + flags + [f"--sysroot={sysroot}",
             f"-B{os.path.join(TOOLS, 'usr', 'lib', 'gcc-cross', 'riscv64-linux-gnu', '13')}",
             f"-B{os.path.join(sysroot, 'lib')}", f"-L{os.path.join(sysroot, 'lib')}",
             "-o", exe, os.path.join(GC, "bench_main.c"), os.path.join(GC, "snn_golden.c"), "-lm"], env=env)
    if c.returncode != 0:
        return {"available": False, "compile_error": c.stderr[-2000:]}
    res = {}
    for reps in (2, 3):
        cmd = (f"{qemu} -d in_asm,exec,nochain -D /dev/stdout {exe} {mod} {img} {reps} {dmode} {dt_us} 2>/dev/null"
               f" | {sys.executable} {os.path.join(ROOT, 'tools', 'qemu_count.py')}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        m = re.search(r"total insns: (\d+)", r.stdout)
        res[reps] = int(m.group(1)) if m else None
    ok = res[2] is not None and res[3] is not None
    return {"available": ok, "insns_per_run": (res[3] - res[2]) if ok else None, "raw": res,
            "flags": " ".join(flags), "method": "qemu-riscv64 8.2.2 in_asm+exec log counting"}


def main(reps=25):
    run(["make", "-C", GC])
    d = load_seed(0)
    m = d["modules"][50]
    out = {"module": "E1 seed 0, offset 50 (test module)", "query": "first test query (seed 0)"}
    for cfg_name, dmode, dt_us in [("l_hw", 3, 100), ("l_hw_dt05", 3, 500)]:
        cfg = CONFIGS[cfg_name]
        inc = FX.weight_increments(m["W"], 8)
        thr = FX.theta_to_thr(m["theta"])
        mod = os.path.join(ROOT, "cache", f"module_{cfg_name}.bin")
        img = os.path.join(ROOT, "cache", "query0.u8")
        pygolden.write_module(mod, inc, thr, module_delay_seed(cfg, 0, 50, 0))
        d["images"]["test_x"][0].reshape(784).astype(np.uint8).tofile(img)
        r = {"x925_core": X925, "a725_core": A725,
             "x925": bench(X925, reps, dmode, dt_us, mod, img),
             "a725": bench(A725, reps, dmode, dt_us, mod, img)}
        ir2 = callgrind_ir(os.path.join(GC, "snn_golden_bench"), [mod, img, "2", str(dmode), str(dt_us)])
        ir3 = callgrind_ir(os.path.join(GC, "snn_golden_bench"), [mod, img, "3", str(dmode), str(dt_us)])
        r["aarch64_instructions_per_run"] = (ir3 - ir2) if (ir2 and ir3) else None
        r["callgrind_ir"] = {"reps2": ir2, "reps3": ir3}
        r["rv64"] = qemu_rv64_insns(mod, img, dmode, dt_us)
        insn = r["rv64"]["insns_per_run"] if r["rv64"].get("available") else r["aarch64_instructions_per_run"]
        r["u54_estimate"] = {"instructions_used": insn,
                             "instruction_source": "rv64 (qemu)" if r["rv64"].get("available") else "aarch64 proxy (callgrind)",
                             "clock_hz": 600e6,
                             "time_s_ipc_1.0": insn / (1.0 * 600e6) if insn else None,
                             "time_s_ipc_0.5": insn / (0.5 * 600e6) if insn else None}
        out[cfg_name] = r
        print(cfg_name, json.dumps(r, indent=1, default=str)[:1500])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e5_bench.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
