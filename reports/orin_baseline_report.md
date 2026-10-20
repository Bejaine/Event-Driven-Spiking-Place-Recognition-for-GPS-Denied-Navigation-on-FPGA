# SpikeNav baselines on the NVIDIA Jetson Orin Nano: report

Produced 2026-10-19 on a Jetson Orin Nano Developer Kit (Super), to answer the judges' question
"why not just use a Jetson?". It uses the same modules, queries and spike-train seeds as the DGX Spark
baselines (`reports/dgx_spark_baseline_report.md`), so the comparisons are like-for-like. Every number
below is in `results/orin_*.json`. `results/orin_summary.json` keys each number to its source and
label, and lists the DGX value next to it.
Labels: MEASURED = from a run on this board; DERIVED = computed from measured numbers (the formula is
given); ESTIMATE = depends on a stated assumption.

## Summary
- **Portable and bit-exact (MEASURED).** The C99 golden model passes all 80 golden vectors on the
  A78AE (`logs/orin/j1_golden_verify.log`), and all 13 pytest tests pass, including the Brian2
  exact-raster test (`logs/orin/pytest_full.log`). Brian2's output for the E1 test module is
  **identical** to the DGX output: 8,206 spikes, max difference 0 (`results/orin_brian2_timing.json`).
  The hardware-config activity statistics and input traces are identical to the DGX's
  (`results/orin_e4_l_hw.json`).
- **CPU, one A78AE core at 1.728 GHz (MEASURED).** The C golden model takes **51.2 ms per module
  run** (dt = 0.1 ms) and executes 254.5 M instructions (DGX: 253.1 M). That is 2.8× slower than the
  DGX's Cortex-X925 and 1.6× slower than its Cortex-A725. Brian2 takes **2.09 s per image per
  module**.
- **CUDA port of the golden model (MEASURED, our own deployment variant).** It is bit-exact with the
  C model for every run of every batch. One 4-module query takes **28.0 ms** on the GPU, and
  throughput saturates at **737 module runs/s**.
- **Power (MEASURED).** Idle is 5.09–5.11 W at VDD_IN in MAXN_SUPER with the desktop running.
  - One busy CPU core: 6.08 W.
  - 4 cores: 7.63 W.
  - CUDA port, one query at a time: 7.14 W.
  - CUDA port, saturated: 9.78 W.
  - torch tooling engine: 16.1–18.2 W.
- **Energy per 4-module query (DERIVED, above idle / total VDD_IN).**
  - CUDA port, one query at a time: **57 / 200 mJ**.
  - CUDA port, saturated batches: **26 / 55 mJ**.
  - C model on 4 cores in parallel: 152 / 456 mJ.
  - C model on 1 core: 201 / 1,232 mJ.
- **Tooling engine (MEASURED).** The DGX torch engine (float32, scatter-bound) reaches 51.5 module
  runs/s here, against 169 on the GB10. It is memory-bound (VDD_SOC rises to about 5 W) and costs
  about 1 J per query above idle. float16 changes the spike counts (90 against 80 E spikes per run),
  so it is not a valid shortcut.
- **One mode only.** No passwordless sudo, so every number is for **MAXN_SUPER** with the default
  `schedutil` / `nvhost_podgov` governors. There was no power-mode sweep (J7).

## Environment (`reports/environment_orin.md`)
- **Board:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super.
- **Software:** L4T R36.5.2 (`nvidia-l4t-core 36.5.2`, JetPack 6.2.x), Ubuntu 22.04.5, kernel
  5.15.199-tegra.
- **CPU:** 6× Cortex-A78AE, 729.6–1728 MHz, `schedutil` governor. Cores 0–3 form one frequency
  cluster and cores 4–5 another.
- **GPU:** Ampere, 8 SMs (sm_87), 306–1020 MHz, `nvhost_podgov` governor.
- **Memory:** 7.4 GiB unified.
- **Power mode:** `nvpmodel -q` reports **MAXN_SUPER**. sudo needs a password, so **no sudo command
  was run** and clocks were not locked.
- **Clocks during timing:**
  - C single-core runs: core 3 at 1728 MHz (p50; mean 1720 MHz). The perf cycle count, 87.1 M cycles
    in 51.2 ms, confirms an effective 1.70 GHz.
  - GPU runs: 1020 MHz (p50).
- **Toolchain:** gcc 11.4.0 (the DGX used 13.3.0), CUDA 12.6 (nvcc V12.6.68). `perf` is not
  installed but `perf_event_paranoid` = 2, so instructions are counted with `tools/perfcount.c`
  (`perf_event_open`, user space only).
- **Python:** venv on the system Python 3.10.12 (`envs/vprsnn`) with numpy 2.2.6, brian2 2.9.0 and
  matplotlib 3.8.4. Disk was too tight for Miniforge.
- **torch:** 2.11.0 from NVIDIA's JetPack 6 / cu126 index
  (`https://pypi.jetson-ai-lab.io/jp6/cu126`), sha256 verified. `torch.cuda.is_available()` is True
  and a 2048² matmul has max relative error 1.0e-6 (`logs/orin/torch_gpu_check.log`). Docker was
  not used.
- **Telemetry:** INA3221 sysfs rails VDD_IN, VDD_CPU_GPU_CV and VDD_SOC, sampled every 10 ms by
  `scripts/orin_power.py`. The sampler was checked against tegrastats (5.11 W against 5.09–5.13 W).
- **Network:** about 55 KB/s, so all downloads were kept to a minimum. No Nordland images were
  downloaded (see J3).

## Correctness (J1)
| Check | Result | Evidence |
|---|---|---|
| `snn_golden_verify golden_vectors 3` (C99, gcc 11.4 -O3, aarch64) | **ALL PASS** (4 × 20 queries bit-exact) | `logs/orin/j1_golden_verify.log` |
| `pytest -q tests/` | **13 passed** (230 s; DGX 23 s) | `logs/orin/pytest_full.log` |
| CUDA port `snn_cuda verify` | **ALL PASS** (80/80 bit-exact) | `results/orin_cuda.json` |
| CUDA port, every run of every batch (1–4,800) against the C model, dt 0.1 and 0.5 ms | 0 mismatched runs | `results/orin_cuda.json` |
| C benchmark workload | 80 E / 8,597 input / 80 I spikes (DGX identical); dt 0.5: 116 / 8,563 / 116 (identical) | `results/orin_cpu.json` |

## Latency (MEASURED; medians; DGX numbers from `results/e5_bench.json`, `results/e1_brian2_clean_timing.json`, `results/e5_gpu.json`)
| Workload | Orin Nano | DGX X925 | DGX A725 | DGX GB10 |
|---|---|---|---|---|
| C golden model, 1 module run, dt = 0.1 ms (1 core) | **51.2 ms** | 18.2 ms | 32.7 ms | — |
| — same, 80 different runs (`snn_golden_verify`) | 49.8 ms | — | — | — |
| C golden model, 1 module run, dt = 0.5 ms (fails accuracy) | 28.8 ms | 11.4 ms | 18.0 ms | — |
| aarch64 instructions per module run, dt 0.1 / 0.5 | 254.5 M / 97.7 M (IPC 2.92 / 1.98) | 253.1 M / 97.4 M | — | — |
| C model, 4-module query on 4 cores in parallel | 51.2 ms (slowest core) | — | — | — |
| Brian2, per image per module (350 ms present + 150 ms rest) | **2.09 s** (1.195 + 0.891) | 0.257 s | 0.511 s | — |
| **CUDA port**, 1 module run (kernel) | **26.1 ms** | — | — | — |
| **CUDA port**, one 4-module query (kernel / incl. copies) | **28.0 / 28.1 ms** | — | — | — |
| CUDA port, dt = 0.5 ms, 4-module query | 9.7 ms | — | — | — |
| CUDA port, throughput (batch ≥ 24) | **708–737 module runs/s** (2,120 at dt 0.5) | — | — | — |
| torch tooling engine, float32 (batch 600 / 1,200 / 2,400 / 4,800) | 46.4 / 47.8 / 50.1 / **51.5** runs/s | — | — | 153 / **169** / 158 / 142 |

- The C runs are pinned to core 3 (`taskset -c 3`), with the sampler on core 0 and `uptime` load
  below 1.
- Brian2 runs pinned to core 3 with nothing else running. Its first `net.run` (80.7 s) includes
  Cython code generation and is excluded, as on the DGX.
- The CUDA port saturates at 24 resident blocks (3 per SM × 8 SMs). Up to that point, batch latency
  is flat at 26–34 ms.

## Power and energy (power MEASURED, energy DERIVED; `results/orin_summary.json`)
Energy per module run = (P_load − P_idle) / rate ("above idle") and P_load / rate ("total"), on VDD_IN.
The idle baseline comes from idle windows in the same script run, interleaved with the load windows.
| Load (≈ 60 s window) | VDD_IN W | VDD_CPU_GPU_CV W | Module runs/s | mJ per run, above idle / total | mJ per 4-module query, above idle / total |
|---|---|---|---|---|---|
| Idle (C session / CUDA session) | 5.09 / 5.10 | 0.73 / 0.74 | — | — | — |
| C model, 1 core | 6.08 | 1.63 | 19.75 | 50.3 / 308 | 201 / 1,232 (4 runs back to back, 205 ms) |
| C model, 4 cores (1 query = 4 modules in parallel) | 7.63 | 3.04 | 66.96 | 37.9 / 114 | 152 / 456 (51 ms) |
| **CUDA port, batch 4 (1 query per launch)** | 7.14 | 2.49 | 142.9 | **14.2 / 50.0** | **57 / 200** (28 ms) |
| **CUDA port, batch 96 (saturated)** | 9.78 | 4.69 | 713.6 | **6.5 / 13.7** | **26 / 55** |
| torch tooling engine, float32, batch 4,800 | 17.98 | 5.28 | 51.5 | 250 / 349 | 999 / 1,396 |
| torch, float16 variant, batch 4,800 (wrong spike counts) | 18.19 | 5.57 | 60.9 | 215 / 299 | 859 / 1,195 |

- In the torch windows VDD_SOC rises from 1.45 W to 4.8–5.6 W (the memory-bound `index_add_`), and
  GPU temperature reaches 75.7 °C without the clock dropping.
- The CUDA loop includes one CPU core spin-waiting in `cudaDeviceSynchronize`.

## Cross-platform consistency (J6)
- **Activity, `e4_activity.py l_hw --seeds 0 10 20` (1,200 module runs).** Every field of
  `results/orin_e4_l_hw.json` equals the DGX `results/e4_l_hw.json`, and the per-step input-event
  traces (1,200 × 3,500) are bit-identical. The seed-0-only run the brief asked for is in
  `results/orin_e4_l_hw_seed0.json`: 8,451.85 input spikes per query, 2.41 events per step,
  81.3 E spikes per run.
- **Brian2 (J3).** `resultPopVecs100_test_E30.npy` for E1 S0 offset 50 is identical to the DGX
  output; R@1 for this module is 0.96, as on the DGX.
- **Ablation, `e3_run.py` + `e3_evaluate.py` (Poisson seed 0, model seeds 0/10/20).** The evaluation
  for `l_hw` and for `a_float` is **identical** to the DGX's P0 entry in `results/e3_summary.json`:
  every threshold, calibration table, R@1/R@5, SeqSLAM and ensemble value. The Brian2 P0 scoring is
  identical too.

  | Config | R@1, seeds 0 / 10 / 20 | Mean R@1 | Ensemble R@1 | SL4 |
  |---|---|---|---|---|
  | `l_hw` | 0.97 / 0.96 / 0.98 | 0.97 | 0.99 | 1.00 |
  | `a_float` | 0.94 / 0.95 / 0.99 | 0.96 | 0.99 | 1.00 |

  Source: `results/orin_e3_summary.json` (P0 only, so the DGX means over P0–P2 are not reproduced).
  The float engine is identical too, so numpy's RNG and float arithmetic behave the same on both
  machines. Running time on the Orin was 22 min (`l_hw`) and 23 min (`a_float`) for 3 seeds.

## Failures, deviations and caveats
- **One power mode.** MAXN_SUPER only; no sudo, so clocks were not locked and there was no J7 sweep.
  The governors ramp to maximum under load (logged: CPU 1728 MHz, GPU 1020 MHz).
- **Board-level power.** VDD_IN is the whole module's input: CPU, GPU, SOC, DRAM and regulator losses.
  It does not include the carrier board's own consumption beyond the module input. The GNOME desktop
  (gdm) was running throughout and is part of "idle".
- **J3 without Nordland on disk.** The network was too slow (about 55 KB/s, so roughly 3 h for the
  summer-tarball prefix). Instead, VPRSNN's `processImageDataset` was replaced by a loader for the
  DGX's pre-processed `e1/S0/images.npz` test images (`scripts/orin_brian2_timing.py`). Everything
  else (network, PoissonGroup, seeds) is VPRSNN's own test mode, and the output matched the DGX
  spike-for-spike. CPU frequency was not sampled during J3, but the identical pinned single-core C
  load sat at 1728 MHz.
- **Compiler.** gcc 11.4 here against 13.3 on the DGX. The instruction count differs by 0.5%.
- **CUDA port scope.** It implements delay mode 3 (hashed delays, 1 ms binned delivery, i.e. the
  proposed hardware config) with hard reset and xorshift input. It is our own deployment variant, not
  part of the DGX baselines. The torch number is the like-for-like comparison with the GB10.
- **float16 torch.** The float16 variant is an explicitly labelled own variant. Its mean E spikes
  (90.2–90.4) differ from float32 (79.9–80.5), so its throughput is not a valid result.
- **Session interruptions.** The board rebooted twice during the run, between measurement phases.
  Every phase was rerun or resumed from scratch, and no measurement spans a reboot (see the
  `uptime_before` / `uptime_after` fields in the JSONs).

## Proposal-ready claims
1. A Jetson Orin Nano Cortex-A78AE core (1.728 GHz, MAXN_SUPER) runs one SpikeNav module in
   51.2 ms (MEASURED) using 50 mJ above idle and 308 mJ total board input (DERIVED;
   `results/orin_cpu.json`, `results/orin_summary.json`).
2. Our bit-exact CUDA port runs a full 4-module SpikeNav query on the Orin Nano GPU in 28.0 ms
   (MEASURED), at 57 mJ above idle and 200 mJ total (DERIVED; `results/orin_cuda.json`).
3. Fully batched, the Orin Nano GPU reaches 714–737 module runs/s at 9.8 W, i.e. 6.5 mJ per module
   run above idle (MEASURED/DERIVED; `results/orin_cuda.json`).
4. The FPGA cycle model predicts 0.62 ms per module run at 125 MHz (L = 100; DERIVED,
   `results/e6_cyclemodel.json`), 42× lower latency than the Orin Nano GPU's 26.1 ms (MEASURED);
   the FPGA side is a model until measured on the Icicle Kit.
5. Brian2, the reference simulator, takes 2.09 s per image per module on the Orin Nano CPU (MEASURED;
   `results/orin_brian2_timing.json`), so a 4-module query costs about 8.3 s (DERIVED, ×4).
6. The fixed-point golden model gives identical spike counts on DGX Spark (X925/A725), Orin Nano
   A78AE and the Orin GPU (MEASURED; `logs/orin/j1_golden_verify.log`, `results/orin_cuda.json`).

## Artifacts
- **Results:** `results/orin_cpu.json`, `results/orin_cuda.json`, `results/orin_gpu.json`,
  `results/orin_brian2_timing.json`, `results/orin_e4_l_hw.json`, `results/orin_e4_l_hw_seed0.json`,
  `results/orin_e3_summary.json`, `results/orin_summary.json`.
- **Logs:** `logs/orin/` (verifier, pytest, benchmarks, queue, Brian2 raw timing), and power CSVs in
  `logs/orin/power/*.csv.gz` (10 ms samples of rails, clocks and temperatures).
- **Code:**
  - `golden_cuda/` (CUDA port), `tools/perfcount.c` (instruction counter).
  - `scripts/orin_env.sh`, `orin_power.py`, `orin_cpu_bench.py`, `orin_gpu.py`,
    `orin_brian2_timing.py`, `orin_queue.sh`, `orin_summary.py`.
- **Reproduce:**
  1. `make -C golden_c && make -C golden_cuda && gcc -O2 -o tools/perfcount tools/perfcount.c`
  2. `python scripts/orin_cpu_bench.py 3 25`
  3. `scripts/orin_queue.sh`
  4. `python scripts/orin_summary.py`
