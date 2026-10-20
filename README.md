# SpikeNav: Event-Driven Spiking Place Recognition for GPS-Denied Navigation on FPGA

**Team Merconez | PolarFire® FPGA Design Contest 2026 | Track 2: PolarFire SoC Icicle Kit (MPFS250T)**

Robots in tunnels, mines, warehouses and disaster sites lose GPS, and their odometry drifts. Visual
place recognition (VPR) corrects that drift by recognising places the robot has seen before.
**SpikeNav** runs the spiking-neural-network VPR model **VPRSNN** (Hussaini, Milford, Fischer,
IEEE T-RO 2024; [QVPR/VPRSNN](https://github.com/QVPR/VPRSNN), MIT License, see `LICENSE.VPRSNN`)
as a dedicated accelerator in PolarFire SoC fabric. The RISC-V cores run Linux, scoring, sequence
matching and a localiser for GPS-free teach-and-repeat navigation.

Before writing any RTL, we pinned down everything the FPGA has to match or beat. That work is all
in this repository:

1. **What the FPGA must compute:** a bit-exact fixed-point golden model (C99) plus golden test
   vectors.
2. **How accurate the hardware can be:** every hardware approximation ablated against the reference
   simulator, at 100 and 2,700 places.
3. **What the alternatives achieve:** measured baselines on an **NVIDIA DGX Spark** and an
   **NVIDIA Jetson Orin Nano** (the obvious "why not just use a Jetson?" alternative). These include
   our own bit-exact CUDA port and measured power.
4. **What the fabric should achieve:** a cycle model of the planned engine, driven by measured spike
   traces.

## Proposal documents (Round 1)
| Document | File |
|---|---|
| System block diagram + evidence (6-page PDF) | [`proposal/Merconez_Track2_SystemBlockDiagram.pdf`](proposal/Merconez_Track2_SystemBlockDiagram.pdf) |
| Source of the PDF (`python make_block_diagram.py`, needs `reportlab`) | [`proposal/make_block_diagram.py`](proposal/make_block_diagram.py) |
| Portal texts (Overview, Innovation & Impact), number provenance | [`proposal/Merconez_Proposal_Form_Text.md`](proposal/Merconez_Proposal_Form_Text.md) |

## Baselines at a glance
The same trained modules, queries and spike-train seeds are used on every platform. One module run
is 784 inputs × 800 neurons × 3,500 steps of 0.1 ms; a query uses 4 modules.

Labels: **M** = measured, **D** = derived, **E** = estimate.

| Platform | Implementation | Per module run | Per 4-module query | Energy per query (above idle / total) |
|---|---|---|---|---|
| DGX Spark, Cortex-X925 (3.9 GHz) | Brian2 reference | 257 ms (M) | 1.03 s (D) | — (no rail sensors) |
| DGX Spark, Cortex-X925 | C golden model, 1 core | 18.2 ms (M) | 72.8 ms (D) | — |
| DGX Spark, Cortex-A725 | C golden model, 1 core | 32.7 ms (M) | 131 ms (D) | — |
| DGX Spark, GB10 GPU | PyTorch float engine (tooling) | 169 runs/s (M) | — | — |
| Jetson Orin Nano, A78AE (1.73 GHz) | Brian2 reference | 2.09 s (M) | 8.3 s (D) | — |
| Jetson Orin Nano, 1 A78AE core | C golden model | 51.2 ms (M) | 205 ms (D) | 201 / 1,232 mJ (D) |
| Jetson Orin Nano, 4 A78AE cores | C golden model, modules in parallel | 51.2 ms (M) | 51.2 ms (M) | 152 / 456 mJ (D) |
| **Jetson Orin Nano GPU** | **our CUDA port, bit-exact** | **26.1 ms (M)** | **28.0 ms (M)** | **57 / 200 mJ (D)** |
| Jetson Orin Nano GPU, batched | CUDA port, 96 runs per launch | 714 runs/s (M) | — | 26 / 55 mJ (D) |
| Icicle Kit, U54 RV64GC (600 MHz) | C golden model | 0.48–0.97 s (E) | 1.9–3.9 s (E) | to be measured (PAC1934) |
| **Icicle Kit, FPGA fabric (SpikeNav)** | **4 engines × 50 lanes, 125 MHz** | **1.19 ms (D)** | **~1.2 ms (D)** | **target < 20 mJ** |

- **Jetson power.** Measured with the INA3221 VDD_IN rail (whole module) in MAXN_SUPER; idle is
  5.1 W.
- **Bit-identical everywhere.** The golden vectors, the Brian2 output, the activity traces and the
  ablation re-run are identical on the DGX Spark and the Jetson. The CUDA port matches the C model
  on every run.
- **Full reports:**
  - [`reports/dgx_spark_baseline_report.md`](reports/dgx_spark_baseline_report.md)
  - [`reports/orin_baseline_report.md`](reports/orin_baseline_report.md)
  - Indexes of every number: [`reports/results.json`](reports/results.json) and
    [`results/orin_summary.json`](results/orin_summary.json)

## Accuracy of the hardware configuration
| Setting | Hardware config (8-bit weights, hashed 1 ms delays, hard reset, shared xorshift train, integer state) | Reference |
|---|---|---|
| 100 places, 3 seeds (M) | R@1 0.959 ± 0.010 | Brian2 0.954 ± 0.007 |
| 2,700 Nordland places, released weights (M) | R@1 0.544 / 0.543 | float (Brian2-exact) 0.549 / 0.553 |
| Approximations that **fail** (M) | no delays: 0.448; dt = 0.5 ms: 0.338 | → delays are kept and dt stays at 0.1 ms |

Sources: `results/e3_summary.json`, `results/e7_*.json`.

## How the repository is used for the FPGA work
| Comparison on the Icicle Kit | Baseline from this repository | Where |
|---|---|---|
| **Functional correctness** of the fabric engine (and of a U54 software port) | Expected per-neuron spike counts for 4 trained modules × 20 queries: the FPGA must reproduce them **bit-exactly** | `golden_vectors/`, `golden_c/verify_main.c`, `reports/golden_model_spec.md` |
| **Accuracy** of the deployed system | Ablation and full-scale R@1 above | `results/e3_summary.json`, `results/e7_*.json` |
| **Latency and energy** against the alternatives | DGX Spark and Jetson Orin Nano tables above | `results/e5_*.json`, `results/orin_*.json` |
| **Predicted FPGA latency** (for checking the RTL) | Cycle model: 148,490 cycles per module run at 50 lanes (1.19 ms at 125 MHz); 0.62 ms at 100 lanes | `results/e6_cyclemodel.json`, `cyclemodel/` |
| **Sizing assumptions** | 8,452 input spikes per query; 2.41 events per step (max 13); 81 E spikes per run | `results/e4_l_hw.json` |
| **Memory budget** | 316,054 bytes per module; LSRAM 520 of 812 blocks for 4 banks at 50 lanes (+24 for delay rings) | `results/e6_cyclemodel.json` |

### Running the golden model on the Icicle Kit
**On the U54 cores (Linux).** This gives a measured U54 number to replace the estimate:
```bash
cd golden_c && make snn_golden_verify CC=gcc CFLAGS="-O3 -std=c99"
./snn_golden_verify ../golden_vectors 3                # must print "ALL PASS"; median s per module run
taskset -c 1 ./snn_golden_verify ../golden_vectors 3   # pinned to one U54 hart
```
To cross-compile, use `riscv64-linux-gnu-gcc -O3 -std=c99 -static -march=rv64gc`.

**On the FPGA fabric engine:**
1. Load `golden_vectors/module_k.bin` into a weight bank. The format is: `"SNNM"`, int32 version,
   uint32 delay_seed, int32 inc[784×400] (8-bit code × 16), int32 thr[400].
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
| `proposal/` | Round 1 proposal PDF, its generator script, and the portal texts |
| `snnsim/` | Python models: Brian2-exact float engine and fixed-point golden model (`engine.py`), configs, hash and spike generator (`hwrand.py`), Q formats (`fixed.py`), PyTorch engine, VPRSNN-based evaluator |
| `golden_c/` | Portable C99 golden model, benchmark driver, golden-vector verifier, ctypes wrapper |
| `golden_cuda/` | CUDA port of the golden model (Jetson, sm_87), bit-exact against `golden_vectors/` and the C model |
| `golden_vectors/` | Bit-exact test vectors (4 modules × 20 queries) and manifest |
| `cyclemodel/` | FPGA engine cycle, throughput and LSRAM model |
| `scripts/` | One script per DGX experiment (E1–E8), plus the Jetson scripts `orin_*` |
| `tools/` | `perfcount.c` (instruction counter via `perf_event_open`), QEMU instruction counting |
| `tests/` | pytest: C ≡ Python bit-exactness, shared-count inhibition ≡ full matrix, Brian2 raster equality |
| `results/` | Measured and derived results (JSON) and event traces; `orin_*` = Jetson |
| `reports/` | DGX and Jetson reports, golden-model spec, machine environments, results index, figures |
| `logs/` | Verification logs; `logs/orin/` holds the Jetson logs and power samples |

## Reproducing
- **DGX Spark / any Linux machine:**
  1. Create the environment: `conda create -p envs/vprsnn -c conda-forge python=3.11 numpy=2.2
     matplotlib=3.8 scipy opencv brian2 cython scikit-learn seaborn pytest huggingface_hub gdown`.
  2. Clone VPRSNN at commit `6eff92c` into `VPRSNN/`.
  3. Fetch Nordland with `python scripts/fetch_nordland.py spring fall summer`.
  4. Build with `make -C golden_c`.
  5. Run the experiments listed in `reports/dgx_spark_baseline_report.md`.
- **Jetson Orin Nano:**
  1. Build: `make -C golden_c && make -C golden_cuda && gcc -O2 -o tools/perfcount tools/perfcount.c`.
  2. Run: `python scripts/orin_cpu_bench.py 3 25`, then `scripts/orin_queue.sh`, then
     `python scripts/orin_summary.py`.
  3. See `reports/orin_baseline_report.md` for the environment, including the JetPack 6 PyTorch
     wheel.

Datasets and trained weights are **not** redistributed. Nordland comes from HuggingFace
`Somayeh-h/Nordland`, and the released VPRSNN weights from the Google Drive link in VPRSNN's
`ens_seq/README.md`. The trained modules in `golden_vectors/` are derived from VPRSNN (MIT License).

## Attribution and AI disclosure
SpikeNav builds on VPRSNN (Hussaini, Milford, Fischer, IEEE T-RO 2024, MIT License), which is based
on Diehl and Cook (2015), and on SeqSLAM (Milford and Wyeth, 2012). The hardware architecture and the
fixed-point, golden, cycle and CUDA models are by Merconez. Claude (Anthropic) assisted with drafting,
and Claude Code implemented the evaluation, baseline and CUDA scripts under the team's direction. All
results are reviewed by the team and reproducible from this repository.
