"""Batched numpy engine model of one VPRSNN Modular-SNN module (784 -> 400 E / 400 I).

Reproduces the test-mode network of VPRSNN non_modular_snn/snn_model.py
(https://github.com/QVPR/VPRSNN, MIT License, (c) 2023 Hussaini, Milford, Fischer)
with Brian2's default schedule inside every dt step:

  1. groups     : refractoriness update, Euler state update of E and I (old values)
  2. thresholds : E: v > theta-72mV and timer > 5 ms and not refractory
                  I: v > -40 mV and not refractory;  inputs: Bernoulli(rate*dt)
  3. synapses   : input spikes pushed into the delay queue, due events delivered
                  (delay 0 arrives in the same step); E->I ge += 10.4 one-to-one;
                  I->E gi += 17 * (#I spikes - own I spike)   (shared-count inhibition)
  4. resets     : E: v = -65 mV, timer = 0 ; I: v = -45 mV

B independent networks are simulated in lock-step (vectorised over the batch).
`arith='float'` uses float64 in volts and mirrors Brian2's Euler expressions;
`arith='fixed'` uses the integer golden model of snnsim/fixed.py.
"""
import numpy as np

from . import fixed as FX
from . import hwrand
from . import params as P


class Net:
    """One trained module: input weights W (784, 400) in [0, 1], theta (400,) in volts."""

    def __init__(self, W, theta, module_id, delay_seed, delays=None):
        self.delays = None if delays is None else np.asarray(delays, np.int32)  # explicit table (tests)
        self.W = np.asarray(W, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.module_id = int(module_id)
        self.delay_seed = int(delay_seed)


def quantize_weights_float(W, bits, scale):
    if bits == 0:
        return W
    levels = (1 << bits) - 1
    if scale == "unit":
        return np.clip(np.round(W * levels), 0, levels) / levels
    wmax = W.max()
    return np.clip(np.round(W / wmax * levels), 0, levels) * wmax / levels


def delay_table(cfg, net):
    """Per-synapse delay parameter (784, 400): steps, or whole ms for the binned mode."""
    max_d = P.steps(P.MAX_DELAY, cfg.dt)
    if cfg.delay_mode == "none":
        return np.zeros((P.N_INPUT, P.N_E), np.int32)
    if cfg.delay_mode == "random" and net.delays is not None:
        return net.delays
    if cfg.delay_mode == "random":
        # reference: delay = rand()*10 ms, rounded to the nearest step (Brian2 spike queue)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([net.delay_seed, 7])))
        u = rng.random((P.N_INPUT, P.N_E))
        return np.floor(u * P.MAX_DELAY / cfg.dt + 0.5).astype(np.int32)
    if cfg.delay_mode == "hash":
        return hwrand.hashed_delay_table(net.delay_seed, max_d + 1)
    if cfg.delay_mode in ("hash_1ms", "hash_1ms_binned"):
        ms = hwrand.hashed_delay_table(net.delay_seed, 11)          # 0..10 ms
        if cfg.delay_mode == "hash_1ms":
            return ms * P.steps(1e-3, cfg.dt)
        return ms
    raise ValueError(cfg.delay_mode)


class Engine:
    def __init__(self, cfg, nets, stats=False, explicit_inhibition=False):
        """explicit_inhibition=True applies I->E through the full 400x400 all-but-self
        matrix (VPRSNN AiAe) instead of the shared spike count (used by tests)."""
        self.explicit_inhibition = explicit_inhibition
        self.cfg = cfg
        self.B = len(nets)
        self.stats_on = stats
        dt = cfg.dt
        self.dt = dt
        self.S = P.n_delay_slots(cfg)
        self.spm = P.steps(1e-3, dt)
        self.ref_e = P.refractory_steps(P.REFRAC_E, dt)
        self.ref_i = P.refractory_steps(P.REFRAC_I, dt)
        self.t_lock = P.timer_lockout_steps(dt)
        self.D = np.stack([delay_table(cfg, n) for n in nets])            # (B, 784, 400)
        if cfg.arith == "float":
            self.W = np.stack([quantize_weights_float(n.W, cfg.weight_bits, cfg.weight_scale) for n in nets])
            self.thr_e = np.stack([n.theta - P.OFFSET_E + P.V_THRESH_E for n in nets])
            self.dtype = np.float64
        else:
            self.K = FX.FixedConsts(dt)
            self.W = np.stack([FX.weight_increments(n.W, cfg.weight_bits, cfg.weight_scale) for n in nets])
            self.thr_e = np.stack([FX.theta_to_thr(n.theta) for n in nets])
            self.dtype = np.int64
        self.jidx = np.arange(P.N_E)
        self.reset_state(initial=True)

    # ------------------------------------------------------------------ state
    def reset_state(self, initial):
        """initial=True: Brian2 start-of-run state (v = v_rest-40 mV, timer = 0).
        initial=False: hardware hard reset (v = v_rest, conductances 0, counters ready)."""
        B, N = self.B, P.N_E
        big = 1 << 30
        if self.cfg.arith == "float":
            ve0 = P.V_REST_E - 40e-3 if initial else P.V_REST_E
            vi0 = P.V_REST_I - 40e-3 if initial else P.V_REST_I
        else:
            ve0 = self.K.V_INIT_E if initial else self.K.V_REST_E
            vi0 = self.K.V_INIT_I if initial else self.K.V_REST_I
        self.ve = np.full((B, N), ve0, self.dtype)
        self.vi = np.full((B, N), vi0, self.dtype)
        self.gee = np.zeros((B, N), self.dtype)
        self.gie = np.zeros((B, N), self.dtype)
        self.gei = np.zeros((B, N), self.dtype)
        # steps since last spike (refractoriness), and the E 'timer' in steps
        self.since_e = np.full((B, N), big, np.int64)
        self.since_i = np.full((B, N), big, np.int64)
        self.timer = np.zeros((B, N), np.int64) if initial else np.full((B, N), big, np.int64)
        self.buf = np.zeros((B, self.S, N), self.dtype)
        self.t = 0          # global step counter (ring-buffer index)

    # ------------------------------------------------------------------ one step
    def step(self, ev_b=None, ev_i=None):
        cfg = self.cfg
        # 1. groups: refractoriness then Euler update (all from old values)
        self.since_e += 1
        self.since_i += 1
        self.timer += 1
        nre = self.since_e >= self.ref_e
        nri = self.since_i >= self.ref_i
        if cfg.arith == "float":
            dt = self.dt
            ve, gee, gie = self.ve, self.gee, self.gie
            ve_new = dt * ((P.V_REST_E - ve) + gee * (-ve) + gie * (P.E_INH_E - ve)) / P.TC_E + ve
            self.ve = np.where(nre, ve_new, ve)
            self.gee = -dt * gee / P.TC_GE + gee
            self.gie = -dt * gie / P.TC_GI + gie
            vi, gei = self.vi, self.gei
            vi_new = dt * ((P.V_REST_I - vi) + gei * (-vi)) / P.TC_I + vi
            self.vi = np.where(nri, vi_new, vi)
            self.gei = -dt * gei / P.TC_GE + gei
            spk_i = (self.vi > P.V_THRESH_I) & nri
        else:
            K = self.K
            ve, gee, gie = self.ve, self.gee, self.gie
            C = gie * ((K.E_INH_E - ve) >> FX.VS) - gee * (ve >> FX.VS)
            acc = K.KV_E * (K.V_REST_E - ve) + K.KC_E * C
            ve_new = np.clip(ve + FX.rshr(acc, FX.SH), FX.V_MIN, FX.V_MAX)
            self.ve = np.where(nre, ve_new, ve)
            self.gee = (gee * K.DGE) >> FX.GSH
            self.gie = (gie * K.DGI) >> FX.GSH
            vi, gei = self.vi, self.gei
            acc = K.KV_I * (K.V_REST_I - vi) - K.KC_I * (gei * (vi >> FX.VS))
            vi_new = np.clip(vi + FX.rshr(acc, FX.SH), FX.V_MIN, FX.V_MAX)
            self.vi = np.where(nri, vi_new, vi)
            self.gei = (gei * K.DGE) >> FX.GSH
            spk_i = (self.vi > K.V_TH_I) & nri
        # 2. thresholds
        spk_e = (self.ve > self.thr_e) & (self.timer >= self.t_lock) & nre
        # 3. synapses: input events -> delay queue, deliver due slot, recurrent
        slot_now = self.t % self.S
        if ev_b is not None and len(ev_b):
            d = self.D[ev_b, ev_i]                                    # (K, 400)
            if cfg.delay_mode == "hash_1ms_binned":
                arrive = (self.t // self.spm + 1 + d) * self.spm      # next ms boundaries
            else:
                arrive = self.t + d
            flat = (ev_b[:, None] * self.S + arrive % self.S) * P.N_E + self.jidx[None, :]
            np.add.at(self.buf.reshape(-1), flat.ravel(), self.W[ev_b, ev_i].ravel())
            if self.stats_on:
                np.add.at(self.cnt_buf.reshape(-1), flat.ravel(), 1)
        self.gee = self.gee + self.buf[:, slot_now, :]
        self.buf[:, slot_now, :] = 0
        if self.stats_on:
            occ = self.cnt_buf[:, slot_now, :].max()
            self.max_slot_occ = max(self.max_slot_occ, int(occ))
            self.cnt_buf[:, slot_now, :] = 0
        n_i = spk_i.sum(axis=1, keepdims=True)
        w_ie = P.W_IE if cfg.arith == "float" else self.K.W_IE
        if self.explicit_inhibition:
            M = np.full((P.N_I, P.N_E), w_ie, dtype=self.dtype)
            np.fill_diagonal(M, 0)
            inh = spk_i.astype(self.dtype) @ M
        else:
            inh = w_ie * (n_i - spk_i)                     # shared-count inhibition
        if cfg.arith == "float":
            self.gei = self.gei + P.W_EI * spk_e
            self.gie = self.gie + inh
        else:
            self.gei = np.minimum(self.gei + self.K.W_EI * spk_e, FX.G_MAX)
            self.gie = np.minimum(self.gie + inh, FX.G_MAX)
            self.gee = np.minimum(self.gee, FX.G_MAX)
        # 4. resets
        if cfg.arith == "float":
            self.ve = np.where(spk_e, P.V_RESET_E, self.ve)
            self.vi = np.where(spk_i, P.V_RESET_I, self.vi)
        else:
            self.ve = np.where(spk_e, self.K.V_RESET_E, self.ve)
            self.vi = np.where(spk_i, self.K.V_RESET_I, self.vi)
        self.timer[spk_e] = 0
        self.since_e[spk_e] = 0
        self.since_i[spk_i] = 0
        self.t += 1
        return spk_e, spk_i

    # ------------------------------------------------------------------ runs
    def _stats_init(self):
        self.cnt_buf = np.zeros((self.B, self.S, P.N_E), np.int32)
        self.max_slot_occ = 0
        self.st = {"ev_per_step": [], "e_spk": [], "i_spk": [], "in_spk": [],
                   "v_e": [], "ge_e": [], "gi_e": [], "v_i": [], "ge_i": [],
                   "vmin": np.inf, "vmax": -np.inf, "gemax": 0.0, "gimax": 0.0,
                   "gei_max": 0.0, "vimin": np.inf, "vimax": -np.inf}

    def _sample_state(self, stride_ok):
        st = self.st
        st["vmin"] = min(st["vmin"], float(self.ve.min()))
        st["vmax"] = max(st["vmax"], float(self.ve.max()))
        st["gemax"] = max(st["gemax"], float(self.gee.max()))
        st["gimax"] = max(st["gimax"], float(self.gie.max()))
        st["gei_max"] = max(st["gei_max"], float(self.gei.max()))
        st["vimin"] = min(st["vimin"], float(self.vi.min()))
        st["vimax"] = max(st["vimax"], float(self.vi.max()))
        if stride_ok:
            st["v_e"].append(self.ve[:, ::8].ravel().copy())
            st["ge_e"].append(self.gee[:, ::8].ravel().copy())
            st["gi_e"].append(self.gie[:, ::8].ravel().copy())
            st["v_i"].append(self.vi[:, ::8].ravel().copy())
            st["ge_i"].append(self.gei[:, ::8].ravel().copy())

    def _present(self, trains_q, n_steps, counts):
        """Run one presentation; trains_q[b] = (step, pixel) events; accumulate E spikes."""
        B = self.B
        # merge events of all nets into a step-indexed CSR
        allb, alls, alli = [], [], []
        for b in range(B):
            tr = trains_q[b]
            if len(tr):
                allb.append(np.full(len(tr), b, np.int64))
                alls.append(tr[:, 0].astype(np.int64))
                alli.append(tr[:, 1].astype(np.int64))
        if allb:
            eb, es, ei = np.concatenate(allb), np.concatenate(alls), np.concatenate(alli)
            order = np.argsort(es, kind="stable")
            eb, es, ei = eb[order], es[order], ei[order]
            ptr = np.searchsorted(es, np.arange(n_steps + 1))
        else:
            eb = es = ei = np.zeros(0, np.int64)
            ptr = np.zeros(n_steps + 1, np.int64)
        for s in range(n_steps):
            a, z = ptr[s], ptr[s + 1]
            se, si = self.step(eb[a:z], ei[a:z]) if z > a else self.step()
            counts += se
            if self.stats_on:
                self.st["ev_per_step"].append(np.bincount(eb[a:z], minlength=B) if z > a else np.zeros(B, np.int64))
                self.st["e_spk_step"] += se.sum(1)
                self.st["i_spk_step"] += si.sum(1)
                self._sample_state(s % 50 == 0)

    def _rest(self, n_steps, counts):
        for s in range(n_steps):
            se, si = self.step()
            counts += se
            if self.stats_on:
                self.st["e_spk_step"] += se.sum(1)
                self.st["i_spk_step"] += si.sum(1)
                self._sample_state(False)

    def run_session(self, trains):
        """trains[b][q]: events for net b, query q (all nets share the query count).

        Returns spike counts (B, Q, 400) using the configured counting window.
        carry: 0.5 s initial rest, then per query 350 ms input + 150 ms rest; the count
               for query q covers the rest after q-1 plus the presentation of q (exactly
               as snn_model.py computes current_spike_count).
        hard : per query: hard state reset, 350 ms input; count over the presentation.
        """
        cfg = self.cfg
        Q = len(trains[0])
        n_pres = P.steps(P.T_PRESENT, cfg.dt)
        n_rest = P.steps(P.T_REST, cfg.dt)
        out = np.zeros((self.B, Q, P.N_E), np.int32)
        if self.stats_on:
            self._stats_init()
        if cfg.reset_mode == "carry":
            self.reset_state(initial=True)
            scratch = np.zeros((self.B, P.N_E), np.int64)
            if self.stats_on:
                self.st["e_spk_step"] = np.zeros(self.B, np.int64)
                self.st["i_spk_step"] = np.zeros(self.B, np.int64)
            self._rest(P.steps(P.T_INIT_REST, cfg.dt), scratch)
            counts = np.zeros((self.B, P.N_E), np.int64)
            for q in range(Q):
                if self.stats_on:
                    self.st["e_spk_step"] = np.zeros(self.B, np.int64)
                    self.st["i_spk_step"] = np.zeros(self.B, np.int64)
                self._present([trains[b][q] for b in range(self.B)], n_pres, counts)
                out[:, q] = counts
                counts[:] = 0
                if self.stats_on:
                    self.st["e_spk"].append(self.st["e_spk_step"].copy())
                    self.st["i_spk"].append(self.st["i_spk_step"].copy())
                    self.st["in_spk"].append(np.array([len(trains[b][q]) for b in range(self.B)]))
                self._rest(n_rest, counts)
        elif cfg.reset_mode == "hard":
            for q in range(Q):
                self.reset_state(initial=False)
                if self.stats_on:
                    self.st["e_spk_step"] = np.zeros(self.B, np.int64)
                    self.st["i_spk_step"] = np.zeros(self.B, np.int64)
                counts = np.zeros((self.B, P.N_E), np.int64)
                self._present([trains[b][q] for b in range(self.B)], n_pres, counts)
                out[:, q] = counts
                if self.stats_on:
                    self.st["e_spk"].append(self.st["e_spk_step"].copy())
                    self.st["i_spk"].append(self.st["i_spk_step"].copy())
                    self.st["in_spk"].append(np.array([len(trains[b][q]) for b in range(self.B)]))
        else:
            raise ValueError(cfg.reset_mode)
        return out
