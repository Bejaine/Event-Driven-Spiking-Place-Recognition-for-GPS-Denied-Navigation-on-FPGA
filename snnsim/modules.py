"""Load trained E1 modules (weights, theta, assignments) and cached images from e1/S<seed>."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFSETS = [0, 25, 50, 75, 100, 125]


def load_module(run, off, n_in=784, n_e=400):
    wdir = os.path.join(run, "weights", f"weights_ne400_L25_offset{off}")
    lst = np.load(os.path.join(wdir, "XeAe.npy"), allow_pickle=True)
    W = np.zeros((n_in, n_e))
    W[lst[:, 0].astype(int), lst[:, 1].astype(int)] = lst[:, 2]
    theta = np.load(os.path.join(wdir, "theta_A.npy"))
    odir = os.path.join(run, "outputs", f"outputs_ne400_L25_offset{off}")
    assign = np.load(os.path.join(odir, f"assignments_L25_S8_O{off}.npy"))
    return {"W": W, "theta": theta, "assign": assign}


def load_seed(seed, run=None):
    run = run or os.path.join(ROOT, "e1", f"S{seed}")
    mods = {off: load_module(run, off) for off in OFFSETS}
    imgs = dict(np.load(os.path.join(run, "images.npz")))
    return {"modules": mods, "images": imgs, "run": run}
