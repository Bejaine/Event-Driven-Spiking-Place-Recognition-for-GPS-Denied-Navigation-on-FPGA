"""Deterministic spike-raster equivalence: snnsim float engine vs Brian2.

Builds the VPRSNN test-mode network (equations copied from VPRSNN
non_modular_snn/snn_model.py, MIT License) in Brian2, but drives it with a
SpikeGeneratorGroup carrying a fixed input train and explicit per-synapse delays,
so both simulators see identical inputs. Compares every E and I spike (step, neuron).
Usage: python tests/brian2_equivalence.py [n_queries] [seed]
"""
import os, sys
import numpy as np
import brian2 as b2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snnsim import params as P, engine as E, inputs as I

b2.prefs.codegen.target = os.environ.get("B2_TARGET", "cython")
b2.prefs.codegen.runtime.cython.cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "b2eq")


def brian2_run(W, theta, delays_steps, trains, n_e=400):
    from brian2 import mV, ms, second
    v_rest_e, v_rest_i = -65 * mV, -60 * mV
    v_reset_e, v_reset_i = -65 * mV, -45 * mV
    v_thresh_e, v_thresh_i = -52 * mV, -40 * mV
    refrac_e, refrac_i = 5 * ms, 2 * ms
    tc_e, tc_i, tc_ge, tc_gi = 100 * ms, 10 * ms, 1.0 * ms, 0.5 * ms
    offset = 20.0 * mV
    eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (tc_e)  : volt (unless refractory)
        I_synE = ge * nS * -v               : amp
        I_synI = gi * nS * (-100.*mV-v)     : amp
        dge/dt = -ge/(tc_ge)               : 1
        dgi/dt = -gi/(tc_gi)               : 1
        theta :volt
        dtimer/dt = 0.1  : second'''
    eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (tc_i)  : volt (unless refractory)
        I_synE = ge * nS * -v               : amp
        I_synI = gi * nS * (-85.*mV-v)      : amp
        dge/dt = -ge/(tc_ge)               : 1
        dgi/dt = -gi/(tc_gi)               : 1'''
    ge_ = b2.NeuronGroup(n_e, eqs_e, threshold='(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)',
                         refractory=refrac_e, reset='v = v_reset_e; timer = 0*ms', method='euler', name='e')
    gi_ = b2.NeuronGroup(n_e, eqs_i, threshold='v>v_thresh_i', refractory=refrac_i, reset='v=v_reset_i',
                         method='euler', name='i')
    ge_.v = v_rest_e - 40 * mV
    gi_.v = v_rest_i - 40 * mV
    ge_.theta = theta * b2.volt
    sei = b2.Synapses(ge_, gi_, 'w:1', on_pre='ge_post += w'); sei.connect(j='i'); sei.w = 10.4
    sie = b2.Synapses(gi_, ge_, 'w:1', on_pre='gi_post += w'); sie.connect(condition='i!=j'); sie.w = 17.0
    # input: concatenate presentations (0.5 s initial rest, 350 ms on, 150 ms off)
    n0, npres, nrest = 5000, 3500, 1500
    idx, tms = [], []
    for q, tr in enumerate(trains):
        base = n0 + q * (npres + nrest)
        idx.append(tr[:, 1]); tms.append((base + tr[:, 0]) * 0.1)
    idx = np.concatenate(idx); tms = np.concatenate(tms)
    sg = b2.SpikeGeneratorGroup(784, idx, tms * ms, name='Xe')
    sxe = b2.Synapses(sg, ge_, 'w:1', on_pre='ge_post += w'); sxe.connect(True)
    sxe.w = W[sxe.i[:], sxe.j[:]]
    sxe.delay = delays_steps[sxe.i[:], sxe.j[:]] * 0.1 * ms
    me, mi = b2.SpikeMonitor(ge_), b2.SpikeMonitor(gi_)
    net = b2.Network(ge_, gi_, sei, sie, sg, sxe, me, mi)
    net.run((n0 + len(trains) * (npres + nrest)) * 0.1 * ms)
    se = np.stack([np.round(me.t / (0.1 * ms)).astype(int), me.i[:]], 1)
    si = np.stack([np.round(mi.t / (0.1 * ms)).astype(int), mi.i[:]], 1)
    return se, si


def snnsim_run(W, theta, delays_steps, trains):
    cfg = P.Config(arith="float", reset_mode="carry", delay_mode="random")
    eng = E.Engine(cfg, [E.Net(W, theta, 0, 0, delays=delays_steps)])
    rec_e, rec_i = [], []
    orig = eng.step

    def step(*a):
        se, si = orig(*a)
        t = eng.t - 1
        rec_e.extend((t, j) for j in np.nonzero(se[0])[0])
        rec_i.extend((t, j) for j in np.nonzero(si[0])[0])
        return se, si
    eng.step = step
    eng.run_session([trains])
    return np.array(rec_e).reshape(-1, 2), np.array(rec_i).reshape(-1, 2)


def main(nq=3, seed=0, weights=None):
    rng = np.random.default_rng(seed)
    if weights is None:
        W = rng.random((784, 400)) * 0.3 + 0.003
        W *= 78 / W.sum(0)
        theta = rng.uniform(0.02, 0.035, 400)
        imgs = rng.integers(0, 256, (nq, 784)).astype(np.uint8)
    else:
        W, theta, imgs = weights
    delays = np.floor(rng.random((784, 400)) * 100 + 0.5).astype(np.int32)
    cfg = P.Config()
    trains = [I.ideal_train(np.random.default_rng(seed * 100 + q), imgs[q], 3500, cfg.dt) for q in range(len(imgs))]
    be, bi = brian2_run(W, theta, delays, trains)
    se, si = snnsim_run(W, theta, delays, trains)
    srt = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))] if len(a) else a
    be, bi, se, si = map(srt, (be, bi, se, si))
    eq_e = be.shape == se.shape and np.array_equal(be, se)
    eq_i = bi.shape == si.shape and np.array_equal(bi, si)
    print(f"Brian2 E spikes {len(be)}, I spikes {len(bi)} | snnsim E {len(se)}, I {len(si)}")
    print(f"identical E raster: {eq_e}; identical I raster: {eq_i}")
    if not eq_e and len(be) and len(se):
        n = min(len(be), len(se))
        k = np.argmax(np.any(be[:n] != se[:n], axis=1)) if np.any(be[:n] != se[:n]) else n
        print("first E divergence at index", k, be[max(0,k-2):k+2].tolist(), se[max(0,k-2):k+2].tolist())
    return eq_e and eq_i


if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ok = main(nq, seed)
    sys.exit(0 if ok else 1)
