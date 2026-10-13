"""Fixed-point arithmetic specification of the SpikeNav golden model.

See reports/golden_model_spec.md for the narrative version. Every quantity is an
integer; the comments give the real-valued meaning of one LSB.

  v   (membrane potential, E and I) : int32, LSB = 2^-16 mV, saturating to [V_MIN, V_MAX]
  ge, gi (conductances)             : int32 >= 0, LSB = 1/4080 (= 1/(255*16)), saturating to G_MAX
  weights                           : 8-bit code c, increment = c << 4  (w = c/255)
  thresholds theta                  : per-neuron int32 THR = round((theta[mV] - 72) * 2^16)
  counters (refractory, timer)      : uint16 steps since last spike, saturating at CNT_MAX

Neuron update (E), one step, all from the old state:
  diff_i = (E_INH - v) >> VS              (arithmetic shift, VS = 7  -> LSB 2^-9 mV)
  vq     = v >> VS
  C      = gi * diff_i - ge * vq          (int64)
  acc    = KV * (V_REST - v) + KC * C     (int64)
  v'     = sat(v + rshr(acc, SH))         (rshr = round-half-up arithmetic shift, SH = 30)
  ge'    = (ge * DGE) >> 16 ; gi' = (gi * DGI) >> 16      (floor)
I neurons: same with ge only (no inhibitory input), KV_I, KC_I, V_REST_I.
"""
import math

import numpy as np

from . import params as P

VF = 16                    # v fractional bits (mV)
GU = 4080                  # conductance units per 1.0 (255 * 16)
VS = 7                     # operand shift for the conductance x voltage products
SH = 30                    # accumulator shift of the Euler increment
GSH = 16                   # decay-factor fractional bits
V_MIN = -(1 << 24)         # -256 mV
V_MAX = (1 << 24) - 1
G_MAX = (1 << 23) - 1      # ~2056 in real units
CNT_MAX = 0xFFFF


def rnd(x):
    """Round half up (identical to floor(x + 0.5) in the C model)."""
    return int(math.floor(x + 0.5))


def mv(x_volts):
    return rnd(x_volts * 1e3 * (1 << VF))


def rshr(x, s):
    """Round-half-up arithmetic right shift (works on numpy int64 arrays)."""
    return (x + (1 << (s - 1))) >> s


class FixedConsts:
    """All integer constants for a given step size dt."""

    def __init__(self, dt):
        self.dt = dt
        self.V_REST_E = mv(P.V_REST_E)
        self.V_RESET_E = mv(P.V_RESET_E)
        self.V_REST_I = mv(P.V_REST_I)
        self.V_RESET_I = mv(P.V_RESET_I)
        self.V_TH_I = mv(P.V_THRESH_I)
        self.E_INH_E = mv(P.E_INH_E)
        self.V_INIT_E = mv(P.V_REST_E - 40e-3)
        self.V_INIT_I = mv(P.V_REST_I - 40e-3)
        ke, ki = dt / P.TC_E, dt / P.TC_I
        self.KV_E = rnd((ke * (1 << SH)))
        self.KV_I = rnd((ki * (1 << SH)))
        # C has LSB (1/GU) * 2^-(VF-VS) mV ; increment in v LSBs = k * C * 2^VF / (GU * 2^(VF-VS))
        self.KC_E = rnd((ke * (1 << VS) / GU * (1 << SH)))
        self.KC_I = rnd((ki * (1 << VS) / GU * (1 << SH)))
        self.DGE = rnd(((1 - dt / P.TC_GE) * (1 << GSH)))
        self.DGI = rnd((max(0.0, 1 - dt / P.TC_GI) * (1 << GSH)))
        self.W_EI = rnd((P.W_EI * GU))
        self.W_IE = rnd((P.W_IE * GU))
        self.REF_E = P.refractory_steps(P.REFRAC_E, dt)
        self.REF_I = P.refractory_steps(P.REFRAC_I, dt)
        self.T_LOCK = P.timer_lockout_steps(dt)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k.isupper()}


def theta_to_thr(theta_volts):
    """Per-neuron firing threshold v > theta - 20 mV - 52 mV, in v LSBs."""
    return np.floor((np.asarray(theta_volts) * 1e3 - 72.0) * (1 << VF) + 0.5).astype(np.int64)


def weight_increments(W, bits, scale="unit"):
    """Integer ge increment per synapse (LSB 1/GU) for the given weight quantisation.

    bits=8, unit : c = round(w*255),  inc = c*16          (exact w = c/255)
    bits=4, unit : c = round(w*15),   inc = c*17*16       (exact w = c/15)
    bits=b, max  : c = round(w/wmax*(2^b-1)), inc = round(c * wmax/(2^b-1) * GU)
    bits=0       : inc = round(w*GU)  (12-bit-equivalent, isolates state quantisation)
    """
    W = np.asarray(W, dtype=np.float64)
    if bits == 0:
        return np.round(W * GU).astype(np.int64)
    levels = (1 << bits) - 1
    if scale == "unit":
        c = np.clip(np.round(W * levels), 0, levels)
        return (c * (GU // levels)).astype(np.int64) if GU % levels == 0 else np.round(c / levels * GU).astype(np.int64)
    wmax = W.max()
    c = np.clip(np.round(W / wmax * levels), 0, levels)
    step = int(round(wmax / levels * GU))
    return (c * step).astype(np.int64)
