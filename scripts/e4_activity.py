"""E4: activity statistics for hardware sizing, on the E1 test queries.

Runs the 4 test modules x 100 test queries of each seed (Poisson seed 0) with
statistics enabled, for the float baseline (a_float) and the proposed hardware
configuration (l_hw, and optionally l_hw_dt05). Saves
  results/e4_<cfg>.json            summary statistics (MEASURED)
  results/e4_<cfg>_traces.npz      per-module-run input events per step (cycle model input)
  results/e4_<cfg>_state.npz       sampled state values (for range plots / Q-format choice)
Usage: python scripts/e4_activity.py CONFIG [--seeds 0 10 20]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snnsim import engine as E, fixed as FX, inputs as I, params as P  # noqa: E402
from snnsim.configs import CONFIGS  # noqa: E402
from snnsim.modules import load_seed  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e3_run import TEST_OFFS, module_delay_seed  # noqa: E402


def pct(a, qs=(0, 1, 50, 99, 100)):
    a = np.asarray(a, dtype=np.float64)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean())}


def main(cfg_name, seeds):
    cfg = CONFIGS[cfg_name]
    fixed = cfg.arith == "fixed"
    vs = 1.0 / (1 << FX.VF) if fixed else 1e3        # -> mV
    gs = 1.0 / FX.GU if fixed else 1.0               # -> conductance units
    traces, e_spk, i_spk, in_spk, max_occ = [], [], [], [], 0
    samp = {k: [] for k in ["v_e", "ge_e", "gi_e", "v_i", "ge_i"]}
    ext = {"vmin": np.inf, "vmax": -np.inf, "gemax": 0, "gimax": 0, "gei_max": 0, "vimin": np.inf, "vimax": -np.inf}
    weights, thetas = [], []
    for si, s in enumerate(seeds):
        d = load_seed(s)
        nets = [E.Net(d["modules"][o]["W"], d["modules"][o]["theta"], si * 6 + o // 25,
                      module_delay_seed(cfg, s, o, 0)) for o in TEST_OFFS]
        for o in [0, 25] + TEST_OFFS:
            weights.append(d["modules"][o]["W"].ravel()); thetas.append(d["modules"][o]["theta"])
        imgs = d["images"]["test_x"].reshape(-1, 784)
        trains = I.make_trains(cfg, imgs, P.steps(P.T_PRESENT, cfg.dt), 0, "test",
                               [n.module_id for n in nets], d["images"]["test_place"])
        eng = E.Engine(cfg, nets, stats=True)
        eng.run_session(trains)
        st = eng.st
        ev = np.stack(st["ev_per_step"])                   # (Q*n_pres, B)
        n_pres = P.steps(P.T_PRESENT, cfg.dt)
        ev = ev.reshape(len(imgs), n_pres, len(nets)).transpose(0, 2, 1).reshape(-1, n_pres)
        traces.append(ev.astype(np.int16))
        e_spk.append(np.concatenate(st["e_spk"])); i_spk.append(np.concatenate(st["i_spk"]))
        in_spk.append(np.concatenate(st["in_spk"]))
        max_occ = max(max_occ, eng.max_slot_occ)
        for k in samp:
            samp[k].append(np.concatenate(st[k]))
        for k in ext:
            ext[k] = (min if "min" in k else max)(ext[k], st[k])
        print("seed", s, "done", flush=True)
    tr = np.concatenate(traces)
    W = np.concatenate(weights); th = np.concatenate(thetas)
    res = {
        "config": cfg_name, "seeds": seeds, "n_module_runs": int(tr.shape[0]), "dt": cfg.dt,
        "input_spikes_per_query": pct(np.concatenate(in_spk)),
        "events_per_step": pct(tr.ravel(), (50, 99, 99.9, 100)) | {"max": int(tr.max())},
        "frac_steps_zero_events": float(np.mean(tr == 0)),
        "e_spikes_per_module_run": pct(np.concatenate(e_spk)),
        "i_spikes_per_module_run": pct(np.concatenate(i_spk)),
        "max_events_in_one_delay_slot_per_neuron": int(max_occ),
        "weights": {"n": int(W.size), "max": float(W.max()), "mean": float(W.mean()),
                    "frac_below_1_255": float(np.mean(W < 1 / 255)),
                    "frac_below_half_lsb8": float(np.mean(W < 0.5 / 255)),
                    "frac_zero_after_q8": float(np.mean(np.round(W * 255) == 0)),
                    "frac_zero_after_q4": float(np.mean(np.round(W * 15) == 0)),
                    "hist_edges": np.linspace(0, 1, 41).tolist(),
                    "hist_counts": np.histogram(W, np.linspace(0, 1, 41))[0].tolist()},
        "theta_mV": pct(th * 1e3),
        "v_e_mV": pct(np.concatenate(samp["v_e"]) * vs) | {"min_all": ext["vmin"] * vs, "max_all": ext["vmax"] * vs},
        "ge_e": pct(np.concatenate(samp["ge_e"]) * gs, (50, 99, 99.9, 100)) | {"max_all": ext["gemax"] * gs},
        "gi_e": pct(np.concatenate(samp["gi_e"]) * gs, (50, 99, 99.9, 100)) | {"max_all": ext["gimax"] * gs},
        "v_i_mV": pct(np.concatenate(samp["v_i"]) * vs) | {"min_all": ext["vimin"] * vs, "max_all": ext["vimax"] * vs},
        "ge_i": pct(np.concatenate(samp["ge_i"]) * gs, (50, 99, 99.9, 100)) | {"max_all": ext["gei_max"] * gs},
    }
    if fixed:
        res["int_extremes"] = {k: float(v) for k, v in ext.items()}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"e4_{cfg_name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(ROOT, "results", f"e4_{cfg_name}_traces.npz"), events=tr)
    np.savez_compressed(os.path.join(ROOT, "results", f"e4_{cfg_name}_state.npz"),
                        **{k: np.concatenate(v)[::4] for k, v in samp.items()}, v_scale=vs, g_scale=gs)
    print(json.dumps({k: res[k] for k in ["input_spikes_per_query", "events_per_step", "frac_steps_zero_events",
                                          "e_spikes_per_module_run", "max_events_in_one_delay_slot_per_neuron"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 10, 20])
    a = ap.parse_args()
    main(a.config, a.seeds)
