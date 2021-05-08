import cython

import numpy as np
cimport numpy as np

cdef extern from "evolutionary_tune.h":
    ctypedef packed struct GaussParams:
        float mu
        float sigma
    void predict_cpu_impl(unsigned *fset_lens, GaussParams *fsets, unsigned n,
                          signed char *rules, unsigned rules_len,
                          GaussParams *uxxs, unsigned *ys, unsigned N)
    void tune_lfs_cpu_impl(unsigned *fset_lens, unsigned n, unsigned rules_len,
                           GaussParams *uxxs, unsigned *ys, unsigned N,
                           unsigned population_power, unsigned iterations,
                           GaussParams *fsets, signed char *rules)
    void predict_gpu_impl(unsigned *fset_lens, GaussParams *fsets, unsigned n,
                          signed char *rules, unsigned rules_len,
                          GaussParams *uxxs, unsigned *ys, unsigned N)
    void tune_lfs_gpu_impl(unsigned *fset_lens, unsigned n, unsigned rules_len,
                           GaussParams *uxxs, unsigned *ys, unsigned N,
                           unsigned population_power, unsigned iterations,
                           GaussParams *fsets, signed char *rules)

GaussParamsDtype = np.dtype([('mu', 'f4'), ('sigma', 'f4')])

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_cpu(np.ndarray[unsigned, ndim=1, mode='c'] fset_lens not None,
                np.ndarray[GaussParams, ndim=1, mode='c'] fsets not None,
                np.ndarray[signed char, ndim=2, mode='c'] rules not None,
                np.ndarray[GaussParams, ndim=2, mode='c'] uxxs not None):
    cdef unsigned n, rules_len, N
    n, rules_len, N = rules.shape[1]-1, rules.shape[0], uxxs.shape[0]
    cdef np.ndarray[unsigned, ndim=1, mode='c'] ys = np.zeros(N, dtype=np.uint32)
    predict_cpu_impl(&fset_lens[0], &fsets[0], n,
                     &rules[0, 0], rules_len,
                     &uxxs[0, 0], &ys[0], N)
    return ys


@cython.boundscheck(False)
@cython.wraparound(False)
def tune_lfs_cpu(np.ndarray[unsigned, ndim=1, mode='c'] fset_lens not None,
                 np.ndarray[GaussParams, ndim=2] uxxs not None,
                 np.ndarray[unsigned, ndim=1] ys not None,
                 *, rules_len, population_power, iterations):

    if not uxxs.flags['C_CONTIGUOUS']:
        uxxs = np.ascontiguousarray(uxxs)
    if not ys.flags['C_CONTIGUOUS']:
        ys = np.ascontiguousarray(ys)
    cdef unsigned n, N
    N, n = uxxs.shape[0], uxxs.shape[1]
    cdef np.ndarray[GaussParams, ndim=1, mode="c"] fsets = np.zeros(sum(fset_lens), dtype=GaussParamsDtype)
    cdef np.ndarray[signed char, ndim=2, mode="c"] rules = np.zeros((rules_len, n+1), dtype=np.int8)
    tune_lfs_cpu_impl(&fset_lens[0], n, rules_len, &uxxs[0, 0], &ys[0], N,
                      population_power, iterations, &fsets[0], &rules[0, 0])
    return fsets, rules
            

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_gpu(
    np.ndarray[unsigned, ndim=1, mode='c'] fsets_lens not None,
    np.ndarray[GaussParams, ndim=1, mode='c'] fsets_table not None,
    np.ndarray[signed char, ndim=2, mode='c'] rules not None,
    np.ndarray[GaussParams, ndim=2, mode='c'] uxxs not None):

    cdef unsigned n, rules_len, N
    n, rules_len, N = rules.shape[1]-1, rules.shape[0], uxxs.shape[0]
    cdef np.ndarray[unsigned, ndim=1, mode='c'] ys = np.zeros(N, dtype=np.uint32)
    predict_gpu_impl(&fsets_lens[0], &fsets_table[0], n,
                     &rules[0, 0], rules_len,
                     &uxxs[0, 0], &ys[0], N)
    return ys
        

@cython.boundscheck(False)
@cython.wraparound(False)
def tune_lfs_gpu(
    np.ndarray[unsigned, ndim=1, mode='c'] fsets_lens not None,
    np.ndarray[GaussParams, ndim=2] uxxs not None,
    np.ndarray[unsigned, ndim=1] ys not None,
    *, rules_len, population_power, iterations):

    if not uxxs.flags['C_CONTIGUOUS']:
        uxxs = np.ascontiguousarray(uxxs)
    if not ys.flags['C_CONTIGUOUS']:
        ys = np.ascontiguousarray(ys)
    cdef unsigned N, n
    N, n = uxxs.shape[0], uxxs.shape[1]
    cdef np.ndarray[GaussParams, ndim=1, mode="c"] fsets = np.zeros(sum(fsets_lens), dtype=GaussParamsDtype)
    cdef np.ndarray[signed char, ndim=2, mode="c"] rules = np.zeros((rules_len, n+1), dtype=np.int8)
    tune_lfs_gpu_impl(&fsets_lens[0], n, rules_len,
                      &uxxs[0, 0], &ys[0], N,
                      population_power, iterations,
                      &fsets[0], &rules[0, 0])
    return fsets, rules
