"""J5 (Jetson Orin Nano): sample the INA3221 power rails, CPU/GPU clocks and temperatures.

  sample: python scripts/orin_power.py sample OUT.csv SECONDS [interval_s]
          writes t, P_VDD_IN, P_VDD_CPU_GPU_CV, P_VDD_SOC (mW), per-CPU cur freq (kHz),
          GPU cur freq (Hz), cpu/gpu/tj temperature (mC). Unprivileged (sysfs only).
  summary: python scripts/orin_power.py summary CSV [t0 t1]  -> JSON on stdout

Power = bus voltage (inN_input, mV) x current (currN_input, mA); checked against tegrastats
(5.09 W vs 5.01 W VDD_IN at idle, logs/orin/power_check.log).
"""
import glob
import json
import os
import sys
import time

HW = sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"))[0]
RAILS = {}
for i in (1, 2, 3):
    RAILS[open(f"{HW}/in{i}_label").read().strip()] = i
NCPU = len(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"))
GPU = "/sys/class/devfreq/17000000.gpu/cur_freq"
TZ = {}
for z in glob.glob("/sys/class/thermal/thermal_zone*"):
    name = open(f"{z}/type").read().strip()
    if name in ("cpu-thermal", "gpu-thermal", "tj-thermal"):
        TZ[name] = f"{z}/temp"


def rd(path):
    with open(path) as f:
        return int(f.read())


def sample(out, seconds, interval=0.01):
    names = ["t"] + [f"P_{r}" for r in RAILS] + [f"f_cpu{c}" for c in range(NCPU)] + ["f_gpu"] + [f"T_{n}" for n in TZ]
    t_end = time.monotonic() + seconds
    with open(out, "w") as f:
        f.write(",".join(names) + "\n")
        nxt = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= t_end:
                break
            row = [f"{time.time():.4f}"]
            for i in RAILS.values():
                row.append(f"{rd(f'{HW}/in{i}_input') * rd(f'{HW}/curr{i}_input') / 1000.0:.1f}")
            for c in range(NCPU):
                row.append(str(rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")))
            row.append(str(rd(GPU)))
            for p in TZ.values():
                row.append(str(rd(p)))
            f.write(",".join(row) + "\n")
            nxt += interval
            time.sleep(max(0.0, nxt - time.monotonic()))


def load(csv):
    with open(csv) as f:
        hdr = f.readline().strip().split(",")
        rows = [list(map(float, ln.split(","))) for ln in f if ln.strip()]
    return hdr, rows


def summary(csv, t0=None, t1=None):
    hdr, rows = load(csv)
    if t0 is not None:
        rows = [r for r in rows if t0 <= r[0] <= t1]
    out = {"file": csv, "n_samples": len(rows), "duration_s": rows[-1][0] - rows[0][0] if rows else 0}
    for k, name in enumerate(hdr[1:], 1):
        v = sorted(r[k] for r in rows)
        out[name] = {"mean": sum(v) / len(v), "min": v[0], "max": v[-1], "p50": v[len(v) // 2]}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.01)
    else:
        a = sys.argv[2:]
        print(json.dumps(summary(a[0], *(map(float, a[1:3]) if len(a) > 1 else ())), indent=1))
