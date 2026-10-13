"""Call VPRSNN modular_snn/modular_snn_evaluation.py::main with an argparse Namespace.

The repo script's CLI cannot be used as-is: it registers `--num_query_imgs` twice, which
argparse rejects (ArgumentError: conflicting option string). This wrapper builds the same
Namespace (same defaults as the repo parser; main() overwrites num_query_imgs anyway) and
calls the unmodified main(). Run with the run directory as CWD.
Usage: run_modular_eval.py MODE SEED OFFSET [NUM_CAL NUM_TEST EPOCHS]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))
os.environ.setdefault("TMPDIR", os.path.join(ROOT, "cache"))
from modular_snn.modular_snn_evaluation import main  # noqa: E402

mode, seed, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ncal, ntest, ep = (int(x) for x in (sys.argv[4:7] if len(sys.argv) > 6 else (50, 100, 30)))
main(argparse.Namespace(dataset="nordland", num_labels=25, num_cal_labels=ncal, num_test_labels=ntest,
                        num_query_imgs=ncal + ntest, use_weighted_assignments=False, shuffled=True, skip=8,
                        offset_after_skip=off, folder_id="NRD_SFS", epochs=ep, n_e=400, threshold_i=0,
                        seed=seed, process_mode=mode, ad_path="_offset{}", multi_path="epoch{}_T{}_T{}"))
