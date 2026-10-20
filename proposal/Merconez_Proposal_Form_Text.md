# Merconez | SpikeNav | Round 1 proposal text (paste into the portal)

- **Track:** Track 2 | PolarFire SoC Icicle Kit.
- **Deadline:** October 24, 2026, 11:59 PM PDT, per the official rules PDF. Confirm on the portal, because one Microchip schedule page lists an earlier date.
- **Repository** (linked in both texts and on every PDF page): https://github.com/Bejaine/Event-Driven-Spiking-Place-Recognition-for-GPS-Denied-Navigation-on-FPGA
  **It must be public before submission.** It is currently private, so evaluators would get a 404.

## Project Overview (287 words, limit 300)

```
Problem: Robots in tunnels, mines, warehouses and disaster sites lose GPS, and odometry drifts without bound. Visual place recognition (VPR) corrects drift by recognising known places. The spiking-neural-network model VPRSNN (Hussaini et al., IEEE T-RO 2024) suits low-power robots, but simulates 3,500 neuron time steps per image. We benchmarked it ourselves: one 25-place module takes 18.2 ms on a DGX Spark Arm core and 51 ms on a Jetson Orin Nano core, and our bit-exact CUDA port on the Jetson GPU needs 28 ms and 200 mJ per 4-module query. The Icicle's RISC-V cores alone would need 0.5-1 s per module.

Solution: SpikeNav runs VPRSNN inference in PolarFire SoC fabric. Frames are pre-processed and spike-encoded in fabric. Four parallel engines simulate 784-input, 800-neuron modules with 8-bit LSRAM weights, hash-regenerated synaptic delays, shared-count lateral inhibition and integer neuron dynamics on the math blocks. A RISC-V scheduler streams modules from LPDDR4; Linux fuses ensemble scores, applies sequence matching and combines matches with odometry in a particle filter for GPS-free teach-and-repeat navigation.

Evidence already produced: a spike-exact re-implementation of VPRSNN, a bit-exact C fixed-point golden model with test vectors, and an ablation of every hardware approximation. On 2,700 Nordland places the hardware configuration scores Recall@1 0.54 versus 0.55 for the exact model. All results reproduce bit for bit on the DGX Spark and the Jetson.

Target application: GPS-free route following for inspection, warehouse, mining and search-and-rescue robots.

Expected outcome: board output bit-identical to the golden model; about 1.2 ms per query (cycle model from measured spike traces), 23x faster than the Jetson GPU; over 25 queries/s for a 500-place, three-member ensemble; under 20 mJ per query measured on board; and a live campus teach-and-repeat demo.

Code, baselines and reports: github.com/Bejaine/Event-Driven-Spiking-Place-Recognition-for-GPS-Denied-Navigation-on-FPGA
```

## Innovation & Expected Impact (237 words, limit 250)

```
Neuromorphic place recognition has so far relied on research chips (the VPRSNN authors target Loihi 2) or a dedicated sensor-processor chip with an event camera (LENS, Science Robotics 2025). SpikeNav brings the published, MIT-licensed VPRSNN model to a commercially available SoC FPGA with a standard camera. Our contribution is the hardware, its verification and the navigation system.

Differentiators, each backed by pre-proposal measurements:
1. Accuracy-proven approximations: 8-bit weights, hashed delays, hard reset, a shared spike generator and integer state together cost 1-2 Recall@1 points at 2,700 places.
2. Zero-storage structure: essential synaptic delays are regenerated from a hash; about 160,000 inhibitory synapses collapse into one shared spike count; only input weights occupy LSRAM.
3. Data-driven sizing: measured activity (2.41 input events per 0.1 ms step) drives a cycle model: 1.19 ms per module run on 50-lane engines within the device budget.
4. Map-scale scheduling: modules stream from LPDDR4, and the localiser's belief selects which ones run.
5. Measured against real alternatives: identical workloads on a DGX Spark and a Jetson Orin Nano (Brian2, C, our bit-exact CUDA port, measured power) set the bar: 28 ms and 200 mJ per query on the Jetson GPU.

Impact: real-time, low-power GPS-free localisation for small robots, with ensembles and sequence matching affordable on board. Our golden model, test vectors and cross-platform baselines are already open source, and the engine will follow: a reproducible FPGA path for SNN research and teaching: github.com/Bejaine/Event-Driven-Spiking-Place-Recognition-for-GPS-Denied-Navigation-on-FPGA
```

Word counts split on whitespace, and the URL counts as one word. The margins (13 and 13 words) absorb differences in how the portal counts. If the portal rejects the URL, delete the last line of the Overview (−5 words); the Innovation text and the PDF still carry the link.

## Where each Round 1 judging criterion is answered

The PDF has 6 pages (p1 system diagram, p2 evidence, p3 **baselines**, p4 engine, p5 budget/IP/verification, p6 plan).

| Criterion (weight) | Judging point | Where it is answered |
|---|---|---|
| Technical Feasibility (35%) | Soundness of the architecture | PDF p1 system diagram; p4 engine and VPRSNN-to-hardware mapping; **p2 measured evidence that the approximations preserve accuracy** |
| | Appropriate use of the kit and peripherals | p1 (MSS, FIC, LPDDR4, USB OTG, GbE, PAC1934, mikroBUS / RPi header); p5 LSRAM and math-block budget from the cycle model |
| | Hardware, software, interfaces and IP cores identified | p5 "Hardware, Software and IP Cores" table, including what is already built (golden model, baselines, CUDA port) |
| | Clear implementation approach and plan | p5 verification flow (3 of 5 stages done, reproduced on two machines); p6 schedule, milestones M1–M5, roles |
| | Risks and mitigations | p6 risk table: the two largest risks are retired with data; "an edge GPU is enough" is answered with measurements |
| Innovation & Creativity (25%) | Originality and differentiation | Innovation text (against CPU, Jetson, Loihi 2 plans and LENS); p2 design decisions; **p3 baselines** |
| | Creative use of FPGA features | hashed delays, shared-count inhibition, input-major LSRAM rows, math-block neuron updates, bank modes, localisation-gated scheduling |
| Problem & Solution Clarity (20%) | Clear problem; complete solution | Overview "Problem" (measured DGX, Jetson and U54 times) and "Solution" |
| | Problem-to-FPGA alignment | **p3 table: same workload on DGX Spark, Jetson Orin Nano, U54 and fabric**; p2 time chart across all platforms |
| | Objectives and outcomes | Overview "Expected outcome"; p6 objectives O1–O5 with targets set by the Jetson baseline |
| Potential Impact (20%) | Usefulness, scalability, adoption | Innovation "Impact"; p4 operating modes; p5 module library and training cost; O5 open source (baselines already public) |

## Number provenance

DGX numbers are from `reports/dgx_spark_baseline_report.md` and Jetson numbers from `reports/orin_baseline_report.md`, both in the repository. Labels: M = measured, D = derived, E = estimate.

| Claim | Value | Label | Source (repository) |
|---|---|---|---|
| Float model vs Brian2 | 728 / 728 E spikes identical | M | logs/e2_exact_raster_E1module.log |
| C golden vs Python fixed point | 32,000 / 32,000 counts | M | logs/golden_bitexact_E1.log |
| 100-place R@1, hardware vs Brian2 | 0.959 ± 0.010 vs 0.954 ± 0.007 | M | results/e3_summary.json |
| 2,700-place R@1, float / hardware / own masks | 0.549–0.553 / 0.543–0.544 / 0.532 | M | results/e7_*.json |
| No delays / dt 0.5 ms at 2,700 places | 0.448 / 0.338 | M | results/e7_f_nodelay.json, e7_l_hw_dt05.json |
| Input spikes per image; events per step | 8,452; 2.41 (max 13) | M | results/e4_l_hw.json |
| DGX: Brian2 / C on X925 / C on A725, per module run | 257 / 18.2 / 32.7 ms | M | results/e1_brian2_clean_timing.json, e5_bench.json |
| Jetson: Brian2 / C on one A78AE core, per module run | 2.09 s / 51.2 ms (1.728 GHz, MAXN_SUPER) | M | results/orin_brian2_timing.json, orin_cpu.json |
| Jetson: C on 4 cores, one 4-module query | 51.2 ms; 152 mJ above idle / 456 mJ total | M / D | results/orin_cpu.json, orin_summary.json |
| Jetson GPU, our CUDA port: module run / 4-module query | 26.1 ms / 28.0 ms; 57 mJ above idle / 200 mJ total per query | M / D | results/orin_cuda.json, orin_summary.json |
| Jetson GPU, CUDA port saturated | 714–737 module runs/s at 9.8 W | M | results/orin_cuda.json |
| Jetson GPU, CUDA port, 500 places × 3 members | 713.6 / 60 = 11.9 queries/s | D | results/orin_cuda.json |
| Jetson GPU, PyTorch tooling engine | 51.5 runs/s (DGX GB10: 169) | M | results/orin_gpu.json, e5_gpu.json |
| Cross-platform consistency | golden vectors, Brian2 output (8,206 spikes), activity traces and e3 P0 evaluation identical on DGX and Jetson | M | logs/orin/, results/orin_e4_l_hw.json, orin_e3_summary.json |
| U54 per module run | 0.48–0.97 s (290 M RV64 instructions, 600 MHz, IPC 0.5–1) | E | results/e5_bench.json |
| Fabric module run, 50 / 100 lanes, 125 MHz | 1.19 / 0.62 ms | D | results/e6_cyclemodel.json |
| Fabric speed-up per query vs Jetson GPU / 4 Jetson cores | 28.0 / 1.2 ≈ 23×; 51.2 / 1.2 ≈ 43× | D | above rows |
| Energy target O3 | < 20 mJ per query (≥ 10× below Jetson GPU's 200 mJ), i.e. < 16.7 W at 1.2 ms | D | above rows |
| Streaming, 500 places × 3 members, 50 lanes | 26.4–28.1 queries/s | D | results/e6_cyclemodel.json |
| LSRAM, 50 lanes ×40 + delay rings | 520 + 24 = 544 of 812 | D | results/e6_cyclemodel.json |
| Training 500 places × 3 members | ~31 core-hours, ~3.1 h on 18 cores | D | results/e8_training_projection.json |

Paper Table I (Modular SNN, Nordland FS: 0.42 / 0.55 / 0.67 / 0.78 at SL1/2/4/10) uses a different reference set from the released instance, so the proposal does not compare against it.

## Submission checklist

- [ ] **Make the GitHub repository public** (Settings → General → Danger Zone → Change visibility), then open the link in a private window.
- [ ] Before the deadline, tag the submitted state (`git tag round1-proposal && git push origin round1-proposal`). The rules forbid changing submitted files after the deadline, and the tag gives evaluators a frozen snapshot while `main` keeps evolving.
- [ ] Team lead registers on myMicrochip. Team "Merconez" has 2–3 members, all 18+, and each has viewed the Privacy Notice.
- [ ] Select Track 2 (PolarFire SoC Icicle Kit).
- [ ] Paste both texts above into the portal (English).
- [ ] Upload `Merconez_Track2_SystemBlockDiagram.pdf` (6 pages) to a dedicated public folder (Google Drive / OneDrive / Dropbox, or the repository's `proposal/` folder), named per the portal's conventions. If the portal wants a single-page diagram, page 1 is the system block diagram and pages 2–6 are supporting detail.
- [ ] Open every share link in a private window to confirm it works without login.
- [ ] Do not change anything in the submitted folder after the deadline.
- [ ] Every team member can explain the golden model, the ablation, the cycle model, and the DGX and Jetson baselines. Section 13 requires full technical understanding of AI-assisted work, and the AI disclosure is on PDF p6.
