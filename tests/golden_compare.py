"""Bit-exact comparison: Python fixed-point engine (snnsim) vs C golden model."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "golden_c"))
from snnsim import engine as E, fixed as FX, hwrand, inputs as I, params as P  # noqa: E402
import pygolden  # noqa: E402


def compare(nets, images, cfg, poisson_seed=0, session="test"):
    """Run len(nets) modules x len(images) queries in both models; return (equal, py, c)."""
    Q = len(images)
    mods = [n.module_id for n in nets]
    trains = I.make_trains(cfg, images, P.steps(P.T_PRESENT, cfg.dt), poisson_seed, session, mods, list(range(Q)))
    eng = E.Engine(cfg, nets)
    py = eng.run_session(trains)
    c = np.zeros_like(py)
    for b, n in enumerate(nets):
        g = pygolden.Golden(eng.W[b], eng.thr_e[b], n.delay_seed, cfg.dt, cfg.delay_mode)
        streams = [hwrand.stream_id(poisson_seed, I.SESSION_CODES[session],
                                    I.SHARED_MODULE if cfg.input_sharing == "shared" else n.module_id, q) for q in range(Q)]
        if cfg.reset_mode == "hard":
            for q in range(Q):
                c[b, q], _ = g.run_hard(images[q], streams[q])
        else:
            c[b], _ = g.run_carry(images, streams)
    return np.array_equal(py, c), py, c


def const_check(dt):
    k = FX.FixedConsts(dt).as_dict()
    g = pygolden.Golden(np.zeros((784, 400), np.int32), np.zeros(400, np.int32), 1, dt, "hash")
    ck = g.consts()
    bad = {n: (k[n], ck[n]) for n in k if n in ck and k[n] != ck[n]}
    return bad
