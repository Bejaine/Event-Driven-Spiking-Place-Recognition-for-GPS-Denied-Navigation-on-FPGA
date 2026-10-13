"""Cycle/throughput model of the planned SpikeNav module engine (E6). All outputs DERIVED.

Driven by measured per-step input-event traces (E4, file produced by scripts/e4_activity.py).

Engine model (per module run = one hard-reset presentation of n_steps dt-steps):
  * L parallel lanes; lane l owns neurons {l, l+L, ...}: ceil(400/L) neurons per lane (E and I).
  * Accumulating one input event touches all 400 synapses of that input:
        acc_cycles(t) = events(t) * ceil(400 / L)
  * Neuron update of all E+I neurons in one step:
        upd_cycles = (MUL_E + MUL_I) * ceil(400 / L) / mults_per_lane
    with MUL_E / MUL_I = multiplies per neuron-step of the golden model
    (snnsim/fixed.py: E = gi*diff, ge*vq, KV*(Vr-v), KC*C, ge*DGE, gi*DGI -> 6;
                      I = ge*vq, KC*(..), KV*(Vr-v), ge*DGE -> 4).
  * Accumulation for step t+1 overlaps the update of step t:
        cycles(t) = max(acc_cycles(t), upd_cycles, gen_cycles) + OVERHEAD (2)
    gen_cycles = input spike generation (784 Bernoulli draws per step) by n_gen generators
    each producing `gen_unroll` draws per cycle (golden spec: 16 xorshift32 generators).
"""
import math

import numpy as np

MUL_E = 6
MUL_I = 4
OVERHEAD = 2
N = 400
N_IN = 784


def module_bytes(weight_bits=8, theta_bytes=4, label_bytes=2, mask_bits=1, seed_bytes=4):
    """Bytes streamed from DDR per module (weights + metadata)."""
    w = N_IN * N * weight_bits // 8
    meta = {"theta": N * theta_bytes, "labels": N * label_bytes,
            "mask": math.ceil(N * mask_bits / 8), "delay_seed": seed_bytes}
    return w + sum(meta.values()), w, meta


def cycles_per_run(events_per_step, L, mults_per_lane=2, n_gen=16, gen_unroll=4):
    """Total cycles of one module run given the per-step input-event counts."""
    nl = math.ceil(N / L)
    upd = math.ceil((MUL_E + MUL_I) * nl / mults_per_lane)
    gen = math.ceil(N_IN / (n_gen * gen_unroll))
    acc = np.asarray(events_per_step, dtype=np.int64) * nl
    per_step = np.maximum(np.maximum(acc, upd), gen) + OVERHEAD
    return int(per_step.sum()), {"upd_cycles": upd, "gen_cycles": gen, "lane_neurons": nl,
                                 "frac_steps_acc_bound": float(np.mean(acc > max(upd, gen)))}


def lsram_blocks(L, weight_bits=8):
    """LSRAM (20 Kbit) blocks for one weight bank: depth 784*ceil(400/L) rows x (L*wbits) bits.

    Compared for the x40 (512-deep), x20 (1024-deep) and x10 (2048-deep) aspect ratios.
    """
    depth = N_IN * math.ceil(N / L)
    width = L * weight_bits
    out = {}
    for name, (d, w) in {"x40": (512, 40), "x20": (1024, 20), "x10": (2048, 10)}.items():
        out[name] = math.ceil(depth / d) * math.ceil(width / w)
    return out, depth, width


def ring_buffer_bits(n_slots, g_bits=24):
    """Delay ring buffer storage per engine (n_slots x 400 neurons x g_bits)."""
    return n_slots * N * g_bits


def resident(events_traces, L, f_hz, mults_per_lane=2, engines=4):
    """Resident mode: `engines` engines each hold one module; a query runs all in parallel.
    events_traces: list of per-step event arrays (one per module run). Returns stats."""
    cyc = np.array([cycles_per_run(e, L, mults_per_lane)[0] for e in events_traces])
    t_run = cyc / f_hz
    return {"cycles_mean": float(cyc.mean()), "cycles_p99": float(np.percentile(cyc, 99)),
            "cycles_max": float(cyc.max()), "t_run_mean_s": float(t_run.mean()),
            "t_query_s": float(t_run.mean()),   # 4 modules in parallel on 4 engines
            "t_query_worst_s": float(t_run.max())}


def streaming(t_run_s, bw_bytes_s, bytes_per_module, module_runs_per_query=60, engines=2):
    """Streaming mode: `engines` engines compute while the same number of banks prefetch
    the next modules over a shared DDR link. Per pipeline slot both engines finish one
    module each; the slot lasts max(compute, time to load `engines` modules)."""
    t_load = engines * bytes_per_module / bw_bytes_s
    slot = max(t_run_s, t_load)
    slots = math.ceil(module_runs_per_query / engines)
    t_q = slots * slot
    return {"t_load_slot_s": t_load, "t_slot_s": slot, "t_query_s": t_q, "queries_per_s": 1.0 / t_q,
            "bound": "compute" if t_run_s >= t_load else "ddr"}
