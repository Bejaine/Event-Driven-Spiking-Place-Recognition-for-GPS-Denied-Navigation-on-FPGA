"""Hardware-friendly deterministic randomness: delay hash and xorshift32 spike generator.

Both use only shifts, XORs and 32-bit modular adds (no multipliers) except the
final range reduction, which multiplies a 16-bit value by a small constant
(101 or 11) — implementable as a shift-add. All arithmetic is mod 2^32.
The C golden model (golden_c/snn_golden.c) implements the identical functions.
"""
import numpy as np

M32 = np.uint64(0xFFFFFFFF)


def mix32(x):
    """Multiply-free avalanche mix (Jenkins one-at-a-time final mix), uint32 -> uint32.

    h += h<<10; h ^= h>>6; h += h<<3; h ^= h>>11; h += h<<15   (all mod 2^32)
    Works elementwise on numpy integer arrays (computed in uint64 then masked).
    """
    h = np.asarray(x, dtype=np.uint64) & M32
    h = (h + (h << np.uint64(10))) & M32
    h ^= h >> np.uint64(6)
    h = (h + (h << np.uint64(3))) & M32
    h ^= h >> np.uint64(11)
    h = (h + (h << np.uint64(15))) & M32
    return h


def delay_hash(i, j, seed):
    """32-bit hash of synapse (input i, neuron j) for a module-specific seed.

    key = (i << 9) | j  (i < 1024, j < 512), XOR-ed with the seed, mixed twice.
    """
    key = (np.asarray(i, dtype=np.uint64) << np.uint64(9)) | np.asarray(j, dtype=np.uint64)
    return mix32(mix32(key ^ np.uint64(seed & 0xFFFFFFFF)))


def hashed_delay_table(seed, n_values, n_in=784, n_out=400):
    """Delay value in [0, n_values) for every synapse: ((h >> 16) * n_values) >> 16."""
    i = np.arange(n_in, dtype=np.uint64)[:, None]
    j = np.arange(n_out, dtype=np.uint64)[None, :]
    h = delay_hash(i, j, seed)
    return (((h >> np.uint64(16)) * np.uint64(n_values)) >> np.uint64(16)).astype(np.int32)


# ---------------- xorshift32 Bernoulli spike generator ----------------
N_GEN = 16           # parallel generators; generator g serves pixels g, g+16, ...
THR_BITS = 24        # compare the top 24 bits of the state against a per-pixel threshold


def xorshift32(x):
    x = np.asarray(x, dtype=np.uint64)
    x ^= (x << np.uint64(13)) & M32
    x ^= x >> np.uint64(17)
    x ^= (x << np.uint64(5)) & M32
    return x


def stream_id(poisson_seed, session, module, query):
    """32-bit identifier of one spike train. module = 255 for a train shared by all modules."""
    return ((poisson_seed & 0x3F) << 26) | ((session & 0x3) << 24) | ((module & 0xFF) << 16) | (query & 0xFFFF)


def gen_seeds(stream):
    """Initial (non-zero) states of the N_GEN generators for one spike train."""
    g = np.arange(N_GEN, dtype=np.uint64)
    s = mix32(mix32((np.uint64(stream) * np.uint64(N_GEN) + g) & M32) ^ np.uint64(0xA5A5A5A5))
    s[s == 0] = 1
    return s


def pixel_thresholds(pixels, dt, intensity=4):
    """Integer threshold per pixel: round(rate*dt*2^24), rate = pixel/intensity Hz."""
    p = np.asarray(pixels, dtype=np.float64) / intensity * dt
    return np.floor(p * (1 << THR_BITS) + 0.5).astype(np.int64)


def lfsr_trains(streams, pixel_imgs, n_steps, dt):
    """Generate xorshift32 Bernoulli spike trains for several streams at once.

    streams: (S,) stream ids; pixel_imgs: (S, 784) uint8 images.
    Returns a list of S arrays of (step, pixel) events, sorted by step then pixel.
    Generator g advances once for every pixel it serves (pixel order g, g+16, ...)
    at every step; a spike occurs when (state >> 8) < threshold.
    """
    S = len(streams)
    n_pix = pixel_imgs.shape[1]
    per_gen = n_pix // N_GEN
    assert per_gen * N_GEN == n_pix
    thr = pixel_thresholds(pixel_imgs, dt)                    # (S, 784)
    thr_g = thr.reshape(S, per_gen, N_GEN)                     # [s, k, g] -> pixel k*16+g
    state = np.stack([gen_seeds(s) for s in streams])          # (S, 16)
    events = [[] for _ in range(S)]
    sh = np.uint64(32 - THR_BITS)
    for t in range(n_steps):
        hits = np.zeros((S, per_gen, N_GEN), dtype=bool)
        for k in range(per_gen):
            state = xorshift32(state)
            hits[:, k, :] = (state >> sh).astype(np.int64) < thr_g[:, k, :]
        if hits.any():
            s_idx, k_idx, g_idx = np.nonzero(hits)
            pix = k_idx * N_GEN + g_idx
            for s in np.unique(s_idx):
                m = s_idx == s
                events[s].append(np.stack([np.full(m.sum(), t), np.sort(pix[m])], 1))
    out = []
    for s in range(S):
        out.append(np.concatenate(events[s]).astype(np.int32) if events[s] else np.zeros((0, 2), np.int32))
    return out
