"""Cache the exact pre-processed images each E1 session sees, using the repo's own loader.

For seed S, run dir e1/S<S>: reproduces the processImageDataset calls made by
VPRSNN non_modular_snn/snn_model.py (record: spring traverse, offset 0, 150 images;
calibrate: summer, offset 0, 50 images; test: summer, offset 50, 100 images), with the
same shuffled indices, and saves e1/S<S>/images.npz:
  record_x (150,28,28) uint8, record_place (150,) place index (unshuffled position)
  cal_x, cal_place, test_x, test_place, shuffled_indices
Usage: python scripts/prepare_images.py SEED [NUM_CAL NUM_TEST]
"""
import contextlib
import io
import os
import sys

import numpy as np

os.environ.setdefault("TMPDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
from tools.data_utils import get_train_test_imagenames_path, processImageDataset  # noqa: E402
from tools.snn_model_utils import get_shuffled_indices  # noqa: E402


def load(seed, num_cal=50, num_test=100, run_dir=None):
    run = run_dir or os.path.join(ROOT, "e1", f"S{seed}")
    cwd = os.getcwd()
    os.chdir(run)
    try:
        nq = num_cal + num_test
        with contextlib.redirect_stdout(io.StringIO()):
            sh = get_shuffled_indices(nq, num_cal, shuffled=True, seed=seed)
            train_p, test_p = get_train_test_imagenames_path("nordland", "NRD_SFS", "test")
            _, rec_p = get_train_test_imagenames_path("nordland", "NRD_SFS", "record")
            kw = dict(imWidth=28, imHeight=28, num_patches=7, num_labels=25, num_query_imgs=nq, skip=8,
                      shuffled_indices=sh)
            rec = processImageDataset(rec_p, "test", num_test_labels=nq, offset_after_skip=0, **kw)
            cal = processImageDataset(test_p, "test", num_test_labels=num_cal, offset_after_skip=0, **kw)
            tst = processImageDataset(test_p, "test", num_test_labels=num_test, offset_after_skip=num_cal, **kw)
    finally:
        os.chdir(cwd)
    out = dict(record_x=rec["x"], record_place=sh[:nq], cal_x=cal["x"], cal_place=sh[:num_cal],
               test_x=tst["x"], test_place=sh[num_cal:nq], shuffled_indices=sh,
               record_y=rec["y"].ravel(), cal_y=cal["y"].ravel(), test_y=tst["y"].ravel())
    np.savez_compressed(os.path.join(run, "images.npz"), **out)
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    a = [int(x) for x in sys.argv[2:4]] if len(sys.argv) > 3 else [50, 100]
    o = load(seed, *a)
    print({k: v.shape for k, v in o.items()})
