"""Place-recognition scoring of spike-count matrices with VPRSNN's own evaluation code.

This re-runs the logic of VPRSNN modular_snn/modular_snn_config_evaluation.py::process,
ens_seq/process_ensembles.py and ens_seq/process_seqmatch.py in memory (no plots),
calling the repo's own functions (https://github.com/QVPR/VPRSNN, MIT License,
(c) 2023 Hussaini, Milford, Fischer). `scripts/check_evaluator.py` verifies that it
reproduces the repo scripts' on-disk results bit-for-bit on the Brian2 outputs.

Inputs per module set (one seed):
  record[m]   : (n_record, 400) spike counts on the reference (spring) images
  test[m]     : (n_q, 400) spike counts on the query images
  assign[m]   : (400,) neuron -> place label (from training, repo assignments_*.npy)
"""
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "VPRSNN"))

from non_modular_snn.snn_model_evaluation import (compute_recall, get_recognized_number_ranking,  # noqa: E402
                                                  get_training_neuronal_spikes, get_unshuffled_results,
                                                  invert_dMat)
from modular_snn.modular_snn_config_evaluation import ignore_hyperactive_neurons  # noqa: E402

THRESHOLDS = [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]   # modular_snn_evaluation.py
N_VALUES = [1, 5, 10, 15, 20, 25]


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def summed_rates(record_list, test_list, assign_list, threshold):
    """modular_snn_config_evaluation.process up to summed_rates (shuffled order)."""
    validation = np.concatenate(record_list, axis=1)
    testing = np.concatenate(test_list, axis=1)
    assignments = np.concatenate(assign_list)
    unique = np.unique(assignments)
    neuron_spikes = np.count_nonzero(validation, axis=0)
    testing = ignore_hyperactive_neurons(neuron_spikes, testing, threshold)
    if np.all(testing == 0):
        return None, unique
    sr = np.zeros((len(unique), testing.shape[0]))
    a, b, c, d = get_training_neuronal_spikes(unique, False, [])
    for i in range(testing.shape[0]):
        _, sr[:, i] = get_recognized_number_ranking(assignments, testing[i, :], unique, a, b, c, d, False)
    return sr, unique


def recall_from_summed(sr_unshuffled):
    rates = invert_dMat(sr_unshuffled)
    pred = np.argsort(rates, axis=0)
    n = rates.shape[1]
    with _quiet():
        r = compute_recall(np.arange(n), pred, n, N_VALUES, "", "", allow_save=False)
    return r


def evaluate_split(record_list, test_list, assign_list, shuffled_indices, num_cal, mode, threshold):
    """R@N for one split ('calibrate' or 'test') at one hyperactive threshold.
    Returns (recall dict or None, summed_rates in shuffled order)."""
    sr, unique = summed_rates(record_list, test_list, assign_list, threshold)
    if sr is None:
        return None, None
    sru = get_unshuffled_results(sr, shuffled_indices, num_cal, mode)
    return recall_from_summed(sru), sr


def calibrate_threshold(record_cal, test_cal, assign_cal, shuffled_indices, num_cal):
    """Best threshold on the calibration modules: max R@1, ties -> smallest threshold."""
    best, table = None, {}
    for th in THRESHOLDS:
        r, _ = evaluate_split(record_cal, test_cal, assign_cal, shuffled_indices, num_cal, "calibrate", th)
        table[th] = None if r is None else r[1]
        if r is not None and (best is None or r[1] > table[best]):
            best = th
    return best, table


def ensemble_recall(sr_list, shuffled_list, num_cal):
    """ens_seq/process_ensembles.py: unshuffle each member, sum similarity, R@N."""
    tot = None
    for sr, sh in zip(sr_list, shuffled_list):
        u = get_unshuffled_results(sr, sh, num_cal)
        tot = u if tot is None else tot + u
    return recall_from_summed(tot), tot


def seqslam_recall(sr_unshuffled, seq_len):
    """ens_seq/process_seqmatch.py: diagonal conv of the inverted rates matrix, R@N."""
    from ens_seq.process_seqmatch import compute_dist_matrix_seqslam
    rates = invert_dMat(sr_unshuffled)
    with _quiet():
        dm = compute_dist_matrix_seqslam(rates, seq_len=seq_len)
        pred = np.argsort(dm, axis=0)
        n = dm.shape[0]
        r = compute_recall(np.arange(n), pred, n, N_VALUES, "", "", allow_save=False)
    return r
