"""E7 sanity check: do the released modules cover consecutive (unshuffled) places?

Runs the float engine on spring place images offset..offset+24 (every 8th list entry,
natural order) for released module `offset` and reports within-module R@1 using the
released assignments. High R@1 => unshuffled place order; chance = 4%.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
from tools.data_utils import loadImg, processImage
from snnsim import engine as E, inputs as I, params as P

REL = os.path.join(ROOT, "released_weights", "npy_modularSNN_weights_V2_20231019")


def load_rel(off):
    w = np.load(f"{REL}/weights/weights_ne400_L25_offset{off}/XeAe.npy", allow_pickle=True)
    W = np.zeros((784, 400)); W[w[:, 0].astype(int), w[:, 1].astype(int)] = w[:, 2]
    th = np.load(f"{REL}/weights/weights_ne400_L25_offset{off}/theta_A.npy")
    a = np.load(f"{REL}/outputs/outputs_ne400_L25_offset{off}/assignments_L25_S8_O{off}.npy")
    return W, th, a


def place_images(season, places):
    names = open(os.path.join(ROOT, "VPRSNN", "dataset_imagenames", f"nordland_imageNames_{season}.txt")).read().splitlines()
    return np.stack([processImage(loadImg(os.path.join(ROOT, "data", names[8 * k][2:])), 28, 28, 7).reshape(784) for k in places])


if __name__ == "__main__":
    off = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    W, th, a = load_rel(off)
    places = list(range(off, off + 25))
    imgs = place_images("spring", places)
    cfg = P.Config()
    tr = I.make_trains(cfg, imgs, 3500, 0, "other", [0], places)
    c = E.Engine(cfg, [E.Net(W, th, 0, 1)]).run_session(tr)[0]
    labels = np.arange(off, off + 25)
    score = np.stack([c[:, a == l].sum(1) / max(1, (a == l).sum()) for l in labels], 1)
    pred = labels[score.argmax(1)]
    print(f"module {off}: within-module R@1 on spring places {off}..{off+24} (natural order) = {np.mean(pred == labels):.2f}")
