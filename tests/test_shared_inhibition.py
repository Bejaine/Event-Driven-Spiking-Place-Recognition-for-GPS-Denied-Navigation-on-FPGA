"""pytest: shared-count lateral inhibition == explicit all-but-self I->E synapse matrix.

The hardware adds gi_e[j] += 17 * (N_I_spikes - s_i[j]) instead of applying the
400x400 VPRSNN AiAe matrix (17.0 off-diagonal, 0 on the diagonal). Checks (1) the
identity on random spike vectors (float and integer units) and (2) full engine runs
with the explicit matrix vs the shared count give identical spike counts, for the
float reference and the fixed-point golden model, in hard-reset and carry modes.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from snnsim import engine as E, fixed as FX, inputs as I, params as P  # noqa: E402


def test_identity_random_vectors():
    rng = np.random.default_rng(0)
    for w in [17.0, FX.rnd(17.0 * FX.GU)]:
        M = np.full((400, 400), w); np.fill_diagonal(M, 0)
        for p in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]:
            for _ in range(20):
                s = (rng.random(400) < p).astype(np.int64)
                assert np.array_equal(s @ M, w * (s.sum() - s))


@pytest.mark.parametrize("arith,mode", [("float", "hard"), ("fixed", "hard"), ("float", "carry"), ("fixed", "carry")])
def test_engine_matrix_vs_shared_count(arith, mode):
    rng = np.random.default_rng(3)
    W = rng.random((784, 400)) * 0.3 + 0.003
    W *= 78 / W.sum(0)
    net = E.Net(W, rng.uniform(0.02, 0.035, 400), 0, 5)
    img = rng.integers(0, 256, (2, 784)).astype(np.uint8)
    cfg = P.Config(arith=arith, weight_bits=8 if arith == "fixed" else 0, reset_mode=mode, delay_mode="hash")
    tr = I.make_trains(cfg, img, P.steps(P.T_PRESENT, cfg.dt), 0, "test", [0], [0, 1])
    a = E.Engine(cfg, [net]).run_session(tr)
    b = E.Engine(cfg, [net], explicit_inhibition=True).run_session(tr)
    assert a.sum() > 0
    assert np.array_equal(a, b)
