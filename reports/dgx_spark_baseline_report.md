# SpikeNav baseline experiments on the DGX Spark: final report

This is the report produced on 2026-10-10 on the NVIDIA DGX Spark (GB10). All numbers come from
files in this repository; `reports/results.json` keys every number to its source path.
Labels: MEASURED = from a run; DERIVED = computed from measured numbers; ESTIMATE = depends on a
stated assumption; EXTERNAL = taken from the paper.

## Summary
- **Float model matches Brian2 exactly (MEASURED).** E and I spike rasters are identical
  spike-for-spike on a trained module over 10 queries (728 E spikes).
  Evidence: `logs/e2_exact_raster_E1module.log`.
- **C99 golden model is bit-exact against the Python fixed-point model (MEASURED).** 32,000
  per-neuron counts match (4 trained modules × 20 queries). Evidence: `logs/golden_bitexact_E1.log`.
- **100-place setting (MEASURED).** The proposed hardware configuration (l) scores R@1
  0.959 ± 0.010; Brian2 scores 0.954 ± 0.007 (4 Poisson seeds). This setting saturates, so it can
  only detect large losses. Source: `results/e3_summary.json`.
- **Full Nordland, 2700 test places, released weights (MEASURED).** l scores R@1 0.544 / 0.543
  (Poisson seeds 0/1). The Brian2-exact float model scores 0.549 / 0.553. Removing delays gives 0.448,
  and dt = 0.5 ms gives 0.338. Sources: `results/e7_*.json`.
- **Activity (MEASURED).** 8,452 input spikes per query, 2.41 events per 0.1 ms step (maximum 13),
  and 81 E spikes per module run.
- **CPU baselines (MEASURED).** The C golden model takes 18.2 ms per module run on a Cortex-X925
  and 32.7 ms on a Cortex-A725. Brian2 takes 0.257 s per image per module on an X925.
- **U54 (ESTIMATE).** The C model executes 290 M RV64 instructions per module run. At 600 MHz and
  IPC 0.5–1.0 that is 0.48–0.97 s per module run.
- **FPGA cycle model (DERIVED).** With 100 lanes at 125 MHz a module run takes 0.62 ms. Four 8-bit
  weight banks need 560 of 812 LSRAM blocks.

## Accuracy, E1 (Brian2, 100 test places, MEASURED; `results/e1_brian2_P*.json`)
| | seed 0 | seed 10 | seed 20 | Ensemble of 3 |
|---|---|---|---|---|
| R@1 (P0) | 0.96 | 0.94 | 0.96 | 1.00 |
| R@5 (P0) | 0.99 | 1.00 | 1.00 | 1.00 |
| SeqSLAM L = 2 / 4 / 10 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| R@1 noise band (P0–P3) | 0.93–0.96 | 0.92–0.98 | 0.96–0.99 | 0.99–1.00 |

## Ablation, E3 (100 test places; R@1 = mean over 3 model seeds, ± sd over 3 Poisson seeds)
| Config | R@1 | Δ vs Brian2 (pp) | Ensemble-3 R@1 | SL4 R@1 | Result |
|---|---|---|---|---|---|
| Brian2 | 0.954 ± 0.007 | — | 0.998 | 1.000 | ref |
| (a) float | 0.963 ± 0.003 | +0.9 | 0.993 | 1.000 | Pass |
| (b) 8-bit weights | 0.947 ± 0.015 | −0.8 | 0.997 | 1.000 | Pass |
| (c) 4-bit weights / per-module max scaling | 0.953 / 0.958 | −0.1 / +0.4 | 0.997 / 0.993 | 1.000 | Pass |
| (d) hashed delays, 0.1 ms | 0.957 ± 0.009 | +0.3 | 1.000 | 1.000 | Pass |
| (e) hashed delays, whole ms / 1 ms binned delivery | 0.942 / 0.963 | −1.2 / +0.9 | 0.997 / 0.993 | 0.998 / 1.000 | Pass |
| (f) no delays | 0.944 ± 0.005 | −1.0 | 0.997 | 1.000 | Pass\* |
| (g) dt = 0.2 ms | 0.960 ± 0.007 | +0.6 | 0.997 | 1.000 | Pass |
| (g) dt = 0.5 ms | 0.884 ± 0.017 | −7.0 | 0.987 | 0.998 | **Fail** |
| (h) hard reset | 0.952 ± 0.007 | −0.2 | 0.990 | 1.000 | Pass |
| (i) shared Poisson train | 0.947 ± 0.012 | −0.8 | 0.997 | 1.000 | Pass |
| (j) xorshift spike generator | 0.960 ± 0.012 | +0.6 | 0.997 | 1.000 | Pass |
| (k) fixed-point state | 0.962 ± 0.017 | +0.8 | 1.000 | 1.000 | Pass |
| (l) full hardware config / 4-bit / 4-bit max | 0.959 / 0.962 / 0.963 | +0.5 / +0.8 / +0.9 | 0.993 / 0.997 / 0.993 | 1.000 | Pass |
| (l) full hardware config, dt = 0.5 ms | 0.823 ± 0.023 | −13.1 | 0.970 | 0.997 | **Fail** |

\*(f) passes only because the 100-place test saturates. At 2700 places it costs about 10 pp.

## Full Nordland, E7 (released instance; 600 calibration + 2700 test places; MEASURED)
| Config | R@1 | R@5 | SL2 | SL4 | SL10 |
|---|---|---|---|---|---|
| float (Brian2-exact), P0 / P1 | 0.549 / 0.553 | 0.692 / 0.688 | 0.730 / 0.735 | 0.886 / 0.872 | 0.976 / 0.964 |
| hardware (l), P0 / P1 | 0.544 / 0.543 | 0.692 / 0.691 | 0.714 / 0.719 | 0.856 / 0.861 | 0.966 / 0.969 |
| hardware (l), P0, engine-computed mask | 0.532 | 0.684 | 0.705 | 0.860 | 0.967 |
| no delays | 0.448 | 0.667 | 0.638 | 0.821 | 0.957 |
| hardware at dt = 0.5 ms | 0.338 | 0.594 | 0.549 | 0.753 | 0.928 |

Paper Table I, Modular SNN on "Nordland FS" (EXTERNAL): SL1 0.42, SL2 0.55, SL4 0.67, SL10 0.78.

## Activity, E4 (hardware config, 1,200 module runs, MEASURED)
- **Input spikes per query:** mean 8,452, p50 8,475, p99 8,855.
- **Events per step:** mean 2.41, p99 7, max 13. 8.9% of steps have no events.
- **Spikes per module run:** E mean 82, p99 116, max 143; I spikes equal E spikes.
- **Delay slots:** at most 57 events land in one 1 ms slot for one neuron.
- **Weights:** maximum 0.196; 0.018% are below 1/255; θ ranges over 35.35–39.46 mV.
- **State ranges:** v_E from −97.9 to −28.1 mV; ge_E up to 7.55; gi_E up to 488.

## Baselines, E5 (MEASURED unless marked)
- **Brian2, per image per module:** X925 0.257 s, A725 0.511 s (clean pinned run).
- **C golden model, per module run (dt = 0.1 ms):** X925 18.2 ms, A725 32.7 ms. At dt = 0.5 ms:
  11.4 and 18.0 ms.
- **Instructions per module run:** 253.1 M on aarch64 and 290.1 M on RV64.
- **U54 (ESTIMATE):** 0.48–0.97 s per module run. Measure it with `golden_c/snn_golden_verify`.
- **GPU float engine (tooling only):** up to 169 module-runs per second.

## FPGA cycle model, E6 (DERIVED; `results/e6_cyclemodel.json`)
- **Resident mode, time per module run (= per query):**
  - L = 50: 1.48 / 1.19 / 0.99 ms at 100 / 125 / 150 MHz.
  - L = 100: 0.78 / 0.62 / 0.52 ms.
  - L = 200: 0.53 / 0.42 / 0.35 ms.
- **Streaming mode** (500 places × 3 members, 316,054 bytes per module), L = 100 at 125 MHz:
  26.4 queries/s at 0.5 GB/s, 52.7 at 1 GB/s, 53.6 at 2 GB/s.
- **LSRAM, 4 banks, ×40 blocks:** 520 (L = 50), 560 (L = 100), 640 (L = 200), out of 812.
  1 ms delay rings add 6 blocks per engine.
- **DSP:** L = 100 with 2 multipliers per lane needs 800 multipliers, more than the device's 784
  math blocks. L = 50 (400) fits.

## Caveats
- The 100-place setting saturates.
- E1 used 30 epochs; the released instance used 60.
- The first Brian2 noise-band reruns had a re-seeding bug; they were fixed and rerun.
- The repo's `modular_snn_evaluation.py` CLI has a duplicate argument, so it is called through a
  wrapper (`scripts/run_modular_eval.py`).
- perf was blocked, so instruction counts come from callgrind and qemu log counting.
- E7 uses the Brian2-exact float model as the Brian2 stand-in, with the released Brian2 spring
  responses for the hyperactive mask.
- The paper's Table I value was read through a web summarizer; verify it against the PDF.
