"""pytest: C golden model == Python fixed-point model, bit-exactly (identical per-neuron counts).

Uses the trained E1 test modules (seed 0, 4 test modules) and 20 test queries when
available (e1/S0/images.npz), otherwise synthetic modules. Covers the proposed hardware
configuration (hard reset, shared LFSR train, hashed 1 ms binned delays, 8-bit weights)
and variants (0.1 ms hashed delays, carry-over sessions, 4-bit weights, dt = 0.5 ms).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "golden_c"))
import golden_compare as G  # noqa: E402
from snnsim import engine as E, params as P  # noqa: E402
from snnsim.configs import CONFIGS  # noqa: E402


def _nets_and_images(nq):
    try:
        from snnsim.modules import load_seed
        d = load_seed(0)
        nets = [E.Net(d["modules"][o]["W"], d["modules"][o]["theta"], i, 1000 + o)
                for i, o in enumerate([50, 75, 100, 125])]
        imgs = d["images"]["test_x"][:nq].reshape(nq, 784)
        return nets, imgs, "E1 seed-0 test modules"
    except Exception:
        rng = np.random.default_rng(7)
        nets = []
        for b in range(4):
            W = rng.random((784, 400)) * 0.3 + 0.003
            W *= 78 / W.sum(0)
            nets.append(E.Net(W, rng.uniform(0.02, 0.035, 400), b, 1000 + b))
        return nets, rng.integers(0, 256, (nq, 784)).astype(np.uint8), "synthetic modules"


@pytest.mark.parametrize("name,nq", [("l_hw", 20), ("l_hw_w4", 5), ("l_hw_dt05", 10)])
def test_bit_exact_hw(name, nq):
    nets, imgs, src = _nets_and_images(nq)
    eq, py, c = G.compare(nets, imgs, CONFIGS[name])
    assert py.sum() > 0, "no spikes at all — test would be vacuous"
    assert eq, f"{name}: C != Python ({src}); total py {py.sum()} c {c.sum()}"


def test_bit_exact_carry_hash01():
    nets, imgs, _ = _nets_and_images(4)
    cfg = CONFIGS["l_hw"].with_(reset_mode="carry", delay_mode="hash")
    eq, py, c = G.compare(nets[:2], imgs, cfg)
    assert py.sum() > 0 and eq


@pytest.mark.parametrize("dt", [1e-4, 2e-4, 5e-4])
def test_constants_match(dt):
    assert G.const_check(dt) == {}
