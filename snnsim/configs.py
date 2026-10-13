"""The E3 ablation configurations (a)-(l). Each differs from (a) only where stated."""
from .params import Config

BASE = Config(name="a_float")
HW = dict(arith="fixed", weight_bits=8, delay_mode="hash_1ms_binned", reset_mode="hard",
          input_sharing="shared", rng="lfsr")

CONFIGS = {
    "a_float": BASE,
    "b_w8": BASE.with_(name="b_w8", weight_bits=8),
    "c_w4": BASE.with_(name="c_w4", weight_bits=4),
    "c_w4max": BASE.with_(name="c_w4max", weight_bits=4, weight_scale="max"),
    "d_hash": BASE.with_(name="d_hash", delay_mode="hash"),
    "e_hash1ms": BASE.with_(name="e_hash1ms", delay_mode="hash_1ms"),
    "e_hash1ms_binned": BASE.with_(name="e_hash1ms_binned", delay_mode="hash_1ms_binned"),
    "f_nodelay": BASE.with_(name="f_nodelay", delay_mode="none"),
    "g_dt05": BASE.with_(name="g_dt05", dt=5e-4),
    "g_dt02": BASE.with_(name="g_dt02", dt=2e-4),
    "h_hard": BASE.with_(name="h_hard", reset_mode="hard"),
    "i_shared": BASE.with_(name="i_shared", input_sharing="shared"),
    "j_lfsr": BASE.with_(name="j_lfsr", rng="lfsr"),
    "k_fixed": BASE.with_(name="k_fixed", arith="fixed"),
    "l_hw": BASE.with_(name="l_hw", **HW),
    "l_hw_w4": BASE.with_(name="l_hw_w4", **{**HW, "weight_bits": 4}),
    "l_hw_w4max": BASE.with_(name="l_hw_w4max", **{**HW, "weight_bits": 4, "weight_scale": "max"}),
    "l_hw_dt05": BASE.with_(name="l_hw_dt05", **{**HW, "dt": 5e-4}),
}
