"""Input spike-train generation (Poisson/Bernoulli rate coding of 28x28 images).

Brian2's PoissonGroup emits a spike for input i at a step when rand() < rate_i*dt,
i.e. an independent Bernoulli process per step. `ideal_train` samples exactly that
process (via geometric inter-spike gaps) from numpy's PCG64; `lfsr` trains come from
hwrand.lfsr_trains. A train is an int32 array of (step, pixel) rows sorted by step.
"""
import numpy as np

from . import hwrand
from .params import INTENSITY

SESSION_CODES = {"record": 0, "cal": 1, "test": 2, "other": 3}
SHARED_MODULE = 255


def ideal_train(rng, pixels, n_steps, dt):
    """Bernoulli(p_i) per step for each pixel, p_i = pixel_i / INTENSITY * dt."""
    p = pixels.astype(np.float64) / INTENSITY * dt
    rows = []
    active = np.nonzero(p > 0)[0]
    if len(active) == 0:
        return np.zeros((0, 2), np.int32)
    pa = p[active]
    # number of gaps needed: generous bound, extended if any pixel is not yet past n_steps
    m = int(max(8, np.ceil(n_steps * pa.max() * 1.5 + 10 * np.sqrt(n_steps * pa.max() + 1))))
    gaps = rng.geometric(pa[:, None], size=(len(active), m))
    pos = np.cumsum(gaps, axis=1) - 1          # 0-based step of each spike
    while np.any(pos[:, -1] < n_steps):
        more = rng.geometric(pa[:, None], size=(len(active), m))
        pos = np.concatenate([pos, pos[:, -1:] + np.cumsum(more, axis=1)], axis=1)
    pix_idx, k = np.nonzero(pos < n_steps)
    steps = pos[pix_idx, k]
    ev = np.stack([steps, active[pix_idx]], 1)
    order = np.lexsort((ev[:, 1], ev[:, 0]))
    return ev[order].astype(np.int32)


def make_trains(cfg, images, n_steps, poisson_seed, session, module_ids, query_ids):
    """Spike trains for a batch of networks over a list of queries.

    images: (Q, 784) uint8, one per query; module_ids: (B,) ints; query_ids: (Q,) ints
    (global query identifiers, so the same query gets the same train in every session
    when trains are shared). Returns trains[b][q] arrays.
    With cfg.input_sharing == 'shared' every module sees the same train for a query.
    """
    B, Q = len(module_ids), len(query_ids)
    sess = SESSION_CODES[session]
    shared = cfg.input_sharing == "shared"
    keys = []  # (b, q) -> key
    uniq = {}
    for b in range(B):
        for q in range(Q):
            mod = SHARED_MODULE if shared else int(module_ids[b])
            key = (int(poisson_seed), sess, mod, int(query_ids[q]))
            keys.append(key)
            uniq.setdefault(key, q)
    cache = {}
    if cfg.rng == "ideal":
        for key, q in uniq.items():
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(key))))
            cache[key] = ideal_train(rng, images[q], n_steps, cfg.dt)
    elif cfg.rng == "lfsr":
        klist = list(uniq.keys())
        streams = [hwrand.stream_id(*k) for k in klist]
        imgs = np.stack([images[uniq[k]] for k in klist])
        # chunk to bound memory
        for s0 in range(0, len(klist), 256):
            out = hwrand.lfsr_trains(streams[s0:s0 + 256], imgs[s0:s0 + 256], n_steps, cfg.dt)
            for k, tr in zip(klist[s0:s0 + 256], out):
                cache[k] = tr
    else:
        raise ValueError(cfg.rng)
    trains = [[None] * Q for _ in range(B)]
    it = iter(keys)
    for b in range(B):
        for q in range(Q):
            trains[b][q] = cache[next(it)]
    return trains
