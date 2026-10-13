"""Export bit-exact golden test vectors for the FPGA / Icicle U54 comparison.

Proposed hardware configuration (l_hw): 8-bit weights, hashed 1 ms binned delays, hard reset,
shared xorshift32 input train, fixed-point state, dt = 0.1 ms. Uses the 4 E1 seed-0 test
modules and the first 20 test queries. Expected counts come from the C golden model
(identical to the Python model, see tests/test_golden_c.py).

golden_vectors/
  module_<k>.bin      "SNNM" v1: uint32 delay_seed, int32 inc[784*400], int32 thr[400]
  queries.u8          20 x 784 uint8 pre-processed images (28x28, patch-normalised)
  streams.u32         20 uint32 spike-train stream ids (shared train: module field = 255)
  expected.i32        4 x 20 x 400 int32 E spike counts (module-major)
  manifest.json       all of the above + config + provenance
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "golden_c")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from snnsim import fixed as FX, hwrand, inputs as I
from snnsim.configs import CONFIGS
from snnsim.modules import load_seed
from e3_run import module_delay_seed
import pygolden

OUT = os.path.join(ROOT, "golden_vectors"); os.makedirs(OUT, exist_ok=True)
cfg = CONFIGS["l_hw"]; NQ = 20; OFFS = [50, 75, 100, 125]
d = load_seed(0)
imgs = d["images"]["test_x"][:NQ].reshape(NQ, 784).astype(np.uint8)
places = d["images"]["test_place"][:NQ]
streams = np.array([hwrand.stream_id(0, I.SESSION_CODES["test"], I.SHARED_MODULE, int(p)) for p in places], np.uint32)
exp = np.zeros((len(OFFS), NQ, 400), np.int32); mods = []
for k, o in enumerate(OFFS):
    m = d["modules"][o]
    inc, thr, seed = FX.weight_increments(m["W"], 8), FX.theta_to_thr(m["theta"]), module_delay_seed(cfg, 0, o, 0)
    pygolden.write_module(os.path.join(OUT, f"module_{k}.bin"), inc, thr, seed)
    g = pygolden.Golden(inc, thr, seed, cfg.dt, cfg.delay_mode)
    for q in range(NQ):
        exp[k, q], _ = g.run_hard(imgs[q], int(streams[q]))
    mods.append({"file": f"module_{k}.bin", "e1_seed": 0, "offset": o, "delay_seed": int(seed)})
imgs.tofile(os.path.join(OUT, "queries.u8")); streams.tofile(os.path.join(OUT, "streams.u32"))
exp.tofile(os.path.join(OUT, "expected.i32"))
json.dump({"config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, "dt_us": 100, "delay_mode_c": 3,
           "n_modules": len(OFFS), "n_queries": NQ, "modules": mods, "query_places": [int(p) for p in places],
           "streams": streams.tolist(), "expected_total_e_spikes": int(exp.sum()),
           "fixed_constants_dt0.1ms": FX.FixedConsts(cfg.dt).as_dict()},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("exported", OUT, "total expected E spikes", exp.sum())
