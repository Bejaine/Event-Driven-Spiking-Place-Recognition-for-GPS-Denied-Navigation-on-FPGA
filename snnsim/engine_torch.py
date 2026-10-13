"""Batched PyTorch (GPU) float engine: many independent hard-reset module runs in lock-step.

Same step semantics as snnsim/engine.py (float path, Brian2 schedule), vectorised over
N = modules x queries runs. Input spikes are Bernoulli(rate*dt) drawn on the device.
This is tooling for fast sweeps / throughput measurement, not a deployment baseline.
"""
import torch

from . import params as P


class TorchEngine:
    def __init__(self, W, theta, delays, dt=1e-4, device="cuda", dtype=torch.float32):
        """W: (M, 784, 400) weights; theta: (M, 400) volts; delays: (M, 784, 400) int steps."""
        self.dev, self.dt, self.dtype = device, dt, dtype
        self.W = torch.as_tensor(W, dtype=dtype, device=device)
        self.thr = torch.as_tensor(theta, dtype=dtype, device=device) - P.OFFSET_E + P.V_THRESH_E
        self.D = torch.as_tensor(delays, dtype=torch.int64, device=device)
        self.S = int(self.D.max().item()) + 1
        self.ref_e = P.refractory_steps(P.REFRAC_E, dt)
        self.ref_i = P.refractory_steps(P.REFRAC_I, dt)
        self.t_lock = P.timer_lockout_steps(dt)
        self.j = torch.arange(P.N_E, device=device)

    @torch.no_grad()
    def run_hard(self, module_idx, images, seed=0):
        """module_idx: (N,) module per run; images: (N, 784) uint8. Returns (N, 400) E counts."""
        dev, dt, dt_ = self.dev, self.dt, self.dtype
        g = torch.Generator(device=dev); g.manual_seed(seed)
        mi = torch.as_tensor(module_idx, dtype=torch.int64, device=dev)
        p = torch.as_tensor(images, dtype=dt_, device=dev) / P.INTENSITY * dt
        N, NE = len(mi), P.N_E
        ve = torch.full((N, NE), P.V_REST_E, dtype=dt_, device=dev)
        vi = torch.full((N, NE), P.V_REST_I, dtype=dt_, device=dev)
        gee = torch.zeros((N, NE), dtype=dt_, device=dev); gie = torch.zeros_like(gee); gei = torch.zeros_like(gee)
        big = 1 << 30
        since_e = torch.full((N, NE), big, dtype=torch.int64, device=dev)
        since_i = since_e.clone(); timer = since_e.clone()
        buf = torch.zeros((N * self.S * NE,), dtype=dt_, device=dev)
        thr = self.thr[mi]
        counts = torch.zeros((N, NE), dtype=torch.int32, device=dev)
        for t in range(P.steps(P.T_PRESENT, dt)):
            since_e += 1; since_i += 1; timer += 1
            nre = since_e >= self.ref_e; nri = since_i >= self.ref_i
            ve_new = dt * ((P.V_REST_E - ve) + gee * (-ve) + gie * (P.E_INH_E - ve)) / P.TC_E + ve
            ve = torch.where(nre, ve_new, ve)
            gee = -dt * gee / P.TC_GE + gee
            gie = -dt * gie / P.TC_GI + gie
            vi_new = dt * ((P.V_REST_I - vi) + gei * (-vi)) / P.TC_I + vi
            vi = torch.where(nri, vi_new, vi)
            gei = -dt * gei / P.TC_GE + gei
            spk_i = (vi > P.V_THRESH_I) & nri
            spk_e = (ve > thr) & (timer >= self.t_lock) & nre
            # inputs
            sp = torch.rand((N, P.N_INPUT), generator=g, device=dev, dtype=dt_) < p
            n_idx, i_idx = sp.nonzero(as_tuple=True)
            if len(n_idx):
                m = mi[n_idx]
                arrive = (t + self.D[m, i_idx]) % self.S                          # (K, 400)
                flat = (n_idx[:, None] * self.S + arrive) * NE + self.j[None, :]
                buf.index_add_(0, flat.reshape(-1), self.W[m, i_idx].reshape(-1))
            b3 = buf.view(N, self.S, NE)
            slot = t % self.S
            gee = gee + b3[:, slot, :]
            b3[:, slot, :] = 0
            n_i = spk_i.sum(1, keepdim=True)
            gei = gei + P.W_EI * spk_e
            gie = gie + P.W_IE * (n_i - spk_i.to(dt_))
            ve = torch.where(spk_e, torch.full_like(ve, P.V_RESET_E), ve)
            vi = torch.where(spk_i, torch.full_like(vi, P.V_RESET_I), vi)
            timer = torch.where(spk_e, torch.zeros_like(timer), timer)
            since_e = torch.where(spk_e, torch.zeros_like(since_e), since_e)
            since_i = torch.where(spk_i, torch.zeros_like(since_i), since_i)
            counts += spk_e.to(torch.int32)
        return counts.cpu().numpy()
