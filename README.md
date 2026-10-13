# Event-Driven Spiking Place Recognition for GPS-Denied Navigation on FPGA: DGX Spark baselines

This repository holds the **reference baselines** for SpikeNav, an FPGA accelerator for the
VPRSNN Modular SNN (Hussaini, Milford, Fischer, IEEE T-RO 2024; https://github.com/QVPR/VPRSNN,
MIT License, see `LICENSE.VPRSNN`). The target is the Microchip **PolarFire SoC Icicle Kit**
(MPFS250T).

The numbers here were measured on an NVIDIA DGX Spark (GB10: 10× Cortex-X925, 10× Cortex-A725 and a
Blackwell GPU). They fix three things before any RTL exists:

1. **what the FPGA must compute**: a bit-exact fixed-point golden model;
2. **how accurate the hardware can be**: an ablation of every approximation against Brian2;
3. **how fast the alternatives are**: CPU and GPU baselines, plus a cycle model of the planned engine.

The full experiment report is `reports/dgx_spark_baseline_report.md`. Every number is keyed to its
source file in `reports/results.json`.

## How these numbers are used for the FPGA comparison

| Comparison on the Icicle Kit | Baseline from this repo | Where |
|---|---|---|
| **Functional correctness** of the fabric engine (and of a U54 software port) | Expected per-neuron spike counts for 4 trained modules × 20 queries: the FPGA must reproduce them **bit-exactly** | `golden_vectors/`, `golden_c/verify_main.c`, `reports/golden_model_spec.md` |
| **Accuracy** of the deployed system | R@1 of the proposed hardware config: 0.959 at 100 places (Brian2 0.954 ± 0.007); 0.544 / 0.543 at 2700 places (float reference 0.549 / 0.553) | `results/e3_summary.json`, `results/e7_*.json` |
| **Latency per module run** | C golden model: 18.2 ms on X925, 32.7 ms on A725. Brian2: 0.257 s per image on X925. U54 ESTIMATE: 0.48–0.97 s (290 M RV64 instructions) | `results/e5_bench.json`, `results/e1_brian2_clean_timing.json` |
| **Predicted FPGA latency** (for checking the RTL) | Cycle model: 77,745 cycles per module run at L = 100 (0.62 ms at 125 MHz), driven by measured event traces | `results/e6_cyclemodel.json`, `results/e4_l_hw_traces.npz`, `cyclemodel/` |
| **Sizing assumptions** | 8,452 input spikes per query; 2.41 events per step (max 13); 81 E spikes per run; state ranges for the Q formats | `results/e4_l_hw.json` |
| **Memory budget** | 316,054 bytes per module; LSRAM 560 of 812 blocks for 4 banks at L = 100 | `results/e6_cyclemodel.json` |

### Why this is a good baseline
- **Anchored to the reference implementation.** The float engine reproduces Brian2's spike rasters
  *exactly* (`logs/e2_exact_raster_E1module.log`). Scoring uses VPRSNN's own evaluation code, checked
  against the repo scripts (`logs/check_evaluator.log`).
- **Bit-exact chain to hardware.** Brian2 ≡ float engine, then the per-approximation ablation, then
  the fixed-point Python model ≡ C99 golden model (`logs/golden_bitexact_E1.log`). The RV64 build
  passes the golden vectors too (`logs/golden_verify_rv64_qemu.log`). An FPGA that matches the
  golden vectors therefore inherits the measured accuracy.
- **Honest error bars.** Brian2's own Poisson-seed noise band is measured, there are 3 model seeds and
  3 Poisson seeds per configuration, and a full-scale (2700-place) check is included, because the
  100-place setting saturates.
- **Same workload everywhere.** The same modules, queries and spike-train seeds (`streams.u32`) are
  used on every platform, so latency comparisons are like-for-like.

## Running the comparison on the Icicle Kit

**On the U54 cores (Linux on the Icicle Kit).** This gives a MEASURED U54 number that replaces the
estimate:
```bash
cd golden_c && make snn_golden_verify CC=gcc CFLAGS="-O3 -std=c99"   # native build on the Icicle Kit
./snn_golden_verify ../golden_vectors 3        # must print "ALL PASS"; reports median s per module run
taskset -c 1 ./snn_golden_verify ../golden_vectors 3   # pin to one U54 hart
```
To cross-compile instead, use `riscv64-linux-gnu-gcc -O3 -std=c99 -static -march=rv64gc`.

**On the FPGA fabric engine:**
1. Load `golden_vectors/module_k.bin` into a weight bank. The format is: `"SNNM"`, int32 version,
   uint32 delay_seed, int32 inc[784×400] (equal to 8-bit code × 16), int32 thr[400].
2. For each query q, generate the input train from `streams.u32[q]` with the xorshift32 generators
   (spec §4), or stream the events from the host.
3. Run 3,500 steps (dt = 0.1 ms) after a hard reset, and read back the 400 E spike counts.
4. Compare with `expected.i32` (module-major, 4 × 20 × 400 int32). Count cycles and compare with the
   cycle model.

`reports/golden_model_spec.md` is the RTL contract: number formats, the per-step operation order,
constants for dt = 0.1/0.2/0.5 ms, the delay hash and the spike generator.

## Repository layout
| Path | Contents |
|---|---|
| `snnsim/` | Python models: Brian2-exact float engine and fixed-point golden model (`engine.py`), configs (`configs.py`), hash and spike generator (`hwrand.py`), Q formats (`fixed.py`), GPU engine, VPRSNN-based evaluator |
| `golden_c/` | Portable C99 golden model, benchmark driver, golden-vector verifier, ctypes wrapper |
| `golden_vectors/` | Bit-exact test vectors (4 modules × 20 queries) and manifest |
| `cyclemodel/` | FPGA engine cycle, throughput and LSRAM model |
| `scripts/` | One script per experiment (E1–E8), report and spec generators |
| `tests/` | pytest: C ≡ Python bit-exactness, shared-count inhibition ≡ full matrix, Brian2 raster equality |
| `results/` | MEASURED and DERIVED results (JSON) and event traces |
| `reports/` | Final report, golden-model spec, machine environment, results index, figures |
| `logs/` | Verification logs cited above |

## Reproducing on another machine
1. Create the environment: `conda create -p envs/vprsnn -c conda-forge python=3.11 numpy=2.2
   matplotlib=3.8 scipy opencv brian2 cython scikit-learn seaborn pytest huggingface_hub gdown`.
2. Clone VPRSNN at commit `6eff92c` into `VPRSNN/`.
3. Download Nordland with `python scripts/fetch_nordland.py spring fall summer`.
4. Build the C model with `make -C golden_c`.
5. Run the per-experiment commands listed in `reports/dgx_spark_baseline_report.md` and in the
   script docstrings: E1 `scripts/e1_launch.sh`, E3 `scripts/e3_run.py`, E4 `scripts/e4_activity.py`,
   E5 `scripts/e5_bench.py`, E6 `cyclemodel/run_cyclemodel.py`, E7 `scripts/e7_run.py`.

Datasets and trained weights are **not** redistributed. Sources: Nordland from HuggingFace
`Somayeh-h/Nordland`; released VPRSNN weights from the Google Drive link in VPRSNN's `ens_seq/README.md`.
The trained modules in `golden_vectors/` are derived from VPRSNN (MIT License).
