"""Network constants and simulation configuration for the SpikeNav engine models.

The constants reproduce the test-mode network of VPRSNN
(https://github.com/QVPR/VPRSNN, commit 6eff92c, MIT License,
(c) 2023 Somayeh Hussaini, Michael Milford and Tobias Fischer), specifically
non_modular_snn/snn_model.py. Units: volts and seconds in the float model.
"""
from dataclasses import dataclass, replace

N_INPUT = 784
N_E = 400
N_I = 400

# --- neuron constants (snn_model.py) ---
V_REST_E = -65e-3
V_REST_I = -60e-3
V_RESET_E = -65e-3
V_RESET_I = -45e-3
V_THRESH_E = -52e-3
V_THRESH_I = -40e-3
OFFSET_E = 20e-3          # threshold is v > theta - offset + v_thresh_e
E_INH_E = -100e-3         # inhibitory reversal potential of E cells
E_INH_I = -85e-3          # (unused: I cells receive no inhibition)
REFRAC_E = 5e-3
REFRAC_I = 2e-3
TC_E = 100e-3
TC_I = 10e-3
TC_GE = 1e-3              # args.tc_ge default 1.0 ms
TC_GI = 0.5e-3            # args.tc_gi = 0.5 ms (paper / repo default)
TIMER_RATE = 0.1          # dtimer/dt = 0.1  (timer runs at 1/10 of real time)
W_EI = 10.4               # E->I one-to-one
W_IE = 17.0               # I->E all-but-self
MAX_DELAY = 10e-3         # input->E delays uniform in [0, 10 ms)
INTENSITY = 4             # Poisson rate = pixel / intensity  [Hz]

# --- protocol timing ---
T_INIT_REST = 0.5
T_PRESENT = 0.35
T_REST = 0.15


@dataclass(frozen=True)
class Config:
    """One engine configuration. Each field is one hardware approximation switch."""
    name: str = "float"
    dt: float = 1e-4                 # integration step [s]
    arith: str = "float"             # 'float' (float64) | 'fixed' (integer golden model)
    weight_bits: int = 0             # 0 = full-precision weights; 8 or 4 = uniform codes w=c/(2^b-1)
    weight_scale: str = "unit"       # 'unit': w_q=round(w*(2^b-1))/(2^b-1); 'max': scale by per-module max(w)
    delay_mode: str = "random"       # 'random' | 'hash' | 'hash_1ms' | 'hash_1ms_binned' | 'none'
    reset_mode: str = "carry"        # 'carry' (reference: 0.5 s initial rest, 150 ms rest between images)
                                     # 'hard'  (state reset at the start of every module run)
    input_sharing: str = "independent"  # 'independent' (per-module trains) | 'shared' (one train per query)
    rng: str = "ideal"               # 'ideal' (numpy PCG64 Bernoulli) | 'lfsr' (hardware xorshift32)
    count_window: str = "auto"       # 'auto': repo window for carry (rest+presentation), presentation for hard

    def with_(self, **kw):
        return replace(self, **kw)


def steps(seconds, dt):
    """Number of integration steps in an interval (Brian2 uses integer timesteps)."""
    return int(round(seconds / dt))


def timer_lockout_steps(dt):
    """First k such that k-fold float64 accumulation of TIMER_RATE*dt exceeds REFRAC_E.

    Brian2 integrates timer with Euler (timer += dt*0.1) in float64 and the threshold
    condition is `timer > refrac_e`. For dt = 0.1 ms this is k = 501 (50.1 ms).
    """
    t = 0.0
    k = 0
    while True:
        k += 1
        t = t + dt * TIMER_RATE
        if t > REFRAC_E:
            return k


def refractory_steps(refrac, dt):
    """Brian2: not_refractory = timestep(t - lastspike, dt) >= timestep(refrac, dt)."""
    return int((refrac + 1e-3 * dt) / dt)


def n_delay_slots(cfg):
    """Ring-buffer length (in steps) needed by a delay mode."""
    max_d = steps(MAX_DELAY, cfg.dt)
    if cfg.delay_mode == "hash_1ms_binned":
        return max_d + steps(1e-3, cfg.dt) + 1
    return max_d + 1
