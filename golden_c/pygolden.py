"""ctypes wrapper of golden_c/libsnngolden.so (used by tests and the bench script)."""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DELAY_MODES = {"none": 0, "hash": 1, "hash_1ms": 2, "hash_1ms_binned": 3}


class Consts(C.Structure):
    _fields_ = [("dt", C.c_double)] + [(n, C.c_int64) for n in
               "V_REST_E V_RESET_E V_REST_I V_RESET_I V_TH_I E_INH_E V_INIT_E V_INIT_I KV_E KV_I KC_E KC_I DGE DGI W_EI W_IE".split()] \
        + [(n, C.c_int) for n in "REF_E REF_I T_LOCK steps_per_ms max_delay n_present n_rest n_init_rest".split()]


class Module(C.Structure):
    _fields_ = [("inc", C.POINTER(C.c_int32)), ("thr", C.POINTER(C.c_int32)), ("delay_seed", C.c_uint32)]


class Engine(C.Structure):
    _fields_ = [("m", C.c_void_p), ("k", C.c_void_p), ("delay_mode", C.c_int), ("S", C.c_int)] + \
        [(n, C.c_int64 * 400) for n in ["ve", "vi", "gee", "gie", "gei"]] + \
        [(n, C.c_int32 * 400) for n in ["since_e", "since_i", "timer"]] + \
        [("buf", C.c_void_p), ("delay", C.c_void_p), ("t", C.c_long)]


class Stats(C.Structure):
    _fields_ = [(n, C.c_int64) for n in ["e_spikes", "i_spikes", "in_spikes", "steps"]]


lib = C.CDLL(os.path.join(HERE, "libsnngolden.so"))


class Golden:
    def __init__(self, inc, thr, delay_seed, dt, delay_mode):
        self.inc = np.ascontiguousarray(inc, dtype=np.int32)
        self.thr = np.ascontiguousarray(thr, dtype=np.int32)
        self.k = Consts()
        lib.sg_consts_init(C.byref(self.k), C.c_double(dt))
        self.m = Module(self.inc.ctypes.data_as(C.POINTER(C.c_int32)),
                        self.thr.ctypes.data_as(C.POINTER(C.c_int32)), C.c_uint32(delay_seed & 0xFFFFFFFF))
        self.e = Engine()
        assert lib.sg_engine_init(C.byref(self.e), C.byref(self.m), C.byref(self.k), DELAY_MODES[delay_mode]) == 0

    def consts(self):
        return {n: getattr(self.k, n) for n, _ in Consts._fields_}

    def run_hard(self, img, stream):
        img = np.ascontiguousarray(img, dtype=np.uint8)
        out = np.zeros(400, np.int32); st = Stats()
        lib.sg_run_hard(C.byref(self.e), img.ctypes.data_as(C.c_void_p), C.c_uint32(stream),
                        out.ctypes.data_as(C.c_void_p), C.byref(st))
        return out, st

    def run_carry(self, imgs, streams):
        imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
        streams = np.ascontiguousarray(streams, dtype=np.uint32)
        out = np.zeros((len(imgs), 400), np.int32); st = Stats()
        lib.sg_run_carry(C.byref(self.e), imgs.ctypes.data_as(C.c_void_p), streams.ctypes.data_as(C.c_void_p),
                         C.c_int(len(imgs)), out.ctypes.data_as(C.c_void_p), C.byref(st))
        return out, st

    def __del__(self):
        try:
            lib.sg_engine_free(C.byref(self.e))
        except Exception:
            pass


def write_module(path, inc, thr, delay_seed):
    with open(path, "wb") as f:
        f.write(b"SNNM"); f.write(np.int32(1).tobytes()); f.write(np.uint32(delay_seed).tobytes())
        f.write(np.ascontiguousarray(inc, np.int32).tobytes()); f.write(np.ascontiguousarray(thr, np.int32).tobytes())
