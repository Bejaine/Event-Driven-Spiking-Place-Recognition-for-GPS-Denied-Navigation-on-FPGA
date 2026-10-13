"""E7: full Nordland evaluation of the released VPRSNN Modular SNN instance (Tier 2).

Released instance (Google Drive, npy_modularSNN_weights_V2_20231019): 132 modules x 25
places, 60 epochs, reference = spring + fall, query = summer (repo id NRD_SFS). Its place
order is NOT shuffled (checked by scripts/e7_check_order.py: module X covers places X..X+24).
Protocol (repo defaults for Nordland): skip 8 -> 3300 places; calibration = summer places
0..599 on modules 0..575 (24 modules); test = summer places 600..3299 on modules 600..3275.
Hyperactive masks use reference (spring) responses: released Brian2 record matrices
(resultPopVecs3300.npy) and, for the hardware config, the engine's own spring responses.

Work units (resumable): python scripts/e7_run.py run CONFIG SESSION CHUNK  (8 modules/chunk)
Driver:                 python scripts/e7_run.py all CONFIG [n_procs]
Output: e7/counts/<config>/<session>_c<chunk>.npz  (counts (n_mod, n_q, 400))
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
from snnsim import engine as E, inputs as I, params as P  # noqa: E402
from snnsim.configs import CONFIGS  # noqa: E402

REL = os.path.join(ROOT, "released_weights", "npy_modularSNN_weights_V2_20231019")
OUT = os.path.join(ROOT, "e7")
NPL, NCAL = 3300, 600
CAL_MODS = list(range(0, NCAL, 25))
TEST_MODS = list(range(NCAL, NPL, 25))
ALL_MODS = list(range(0, NPL, 25))
CHUNK = 8
PSEED = int(os.environ.get("E7_PSEED", "0"))     # Poisson seed; output dir gets a _P<p> suffix if != 0


def load_rel(off):
    w = np.load(f"{REL}/weights/weights_ne400_L25_offset{off}/XeAe.npy", allow_pickle=True)
    W = np.zeros((784, 400)); W[w[:, 0].astype(int), w[:, 1].astype(int)] = w[:, 2]
    th = np.load(f"{REL}/weights/weights_ne400_L25_offset{off}/theta_A.npy")
    a = np.load(f"{REL}/outputs/outputs_ne400_L25_offset{off}/assignments_L25_S8_O{off}.npy")
    return W, th, a


def images():
    path = os.path.join(OUT, "images.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    from tools.data_utils import loadImg, processImage
    out = {}
    for season in ["summer", "spring"]:
        names = open(os.path.join(ROOT, "VPRSNN", "dataset_imagenames", f"nordland_imageNames_{season}.txt")).read().splitlines()
        out[season] = np.stack([processImage(loadImg(os.path.join(ROOT, "data", names[8 * k][2:])), 28, 28, 7).reshape(784)
                                for k in range(NPL)]).astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def session_spec(session):
    if session == "cal":
        return CAL_MODS, "summer", list(range(0, NCAL))
    if session == "test":
        return TEST_MODS, "summer", list(range(NCAL, NPL))
    if session == "record":
        return ALL_MODS, "spring", list(range(0, NPL))
    raise ValueError(session)


def n_chunks(session):
    return (len(session_spec(session)[0]) + CHUNK - 1) // CHUNK


def run_unit(cfg_name, session, chunk):
    cfg = CONFIGS[cfg_name]
    tag = cfg_name if PSEED == 0 else f"{cfg_name}_P{PSEED}"
    out = os.path.join(OUT, "counts", tag, f"{session}_c{chunk}.npz")
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mods, season, places = session_spec(session)
    mods = mods[chunk * CHUNK:(chunk + 1) * CHUNK]
    imgs = images()[season][places]
    nets = []
    for off in mods:
        W, th, _ = load_rel(off)
        ds = (7919 * off + 13) & 0xFFFFFFFF
        if cfg.delay_mode == "random":
            ds = ds * 31 + PSEED          # reference random delays are redrawn per seed (as in Brian2)
        nets.append(E.Net(W, th, module_id=off // 25 % 255, delay_seed=ds))
    t0 = time.time()
    tr = I.make_trains(cfg, imgs, P.steps(P.T_PRESENT, cfg.dt), PSEED, "record" if session == "record" else "test",
                       [n.module_id for n in nets], places)
    c = E.Engine(cfg, nets).run_session(tr)
    np.savez_compressed(out, counts=c.astype(np.int16), modules=np.array(mods), wall_s=time.time() - t0)
    print("wrote", out, "%.0fs" % (time.time() - t0), flush=True)


def drive(cfg_name, n_procs, sessions):
    images()
    units = [(s, k) for s in sessions for k in range(n_chunks(s))]
    log = os.path.join(ROOT, "logs", f"e7_{cfg_name}_P{PSEED}.log")

    def go(u):
        with open(log, "a") as f:
            subprocess.run([sys.executable, os.path.abspath(__file__), "run", cfg_name, u[0], str(u[1])],
                           stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1"))
    with ThreadPoolExecutor(n_procs) as ex:
        list(ex.map(go, units))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_unit(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "all":
        cfg = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        sessions = sys.argv[4].split(",") if len(sys.argv) > 4 else ["cal", "test"]
        drive(cfg, n, sessions)
