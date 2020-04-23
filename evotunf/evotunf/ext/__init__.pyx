import cython

import numpy as np
cimport numpy as np

cdef extern from "evotunf_ext.h":
    ctypedef packed struct GaussParams:
        float mu
        float sigma
    void predict_gpu_impl(unsigned *fsets_lens, GaussParams *gauss_params, unsigned char *rules, unsigned rules_len, unsigned n,
                          GaussParams *xxs, unsigned *ys, unsigned N)
    void tune_lfs_gpu_impl(unsigned *fsets_lens, unsigned rules_len, unsigned n, GaussParams *xxs, unsigned *ys, unsigned N,
                           unsigned mu, unsigned lamda, unsigned iterations, GaussParams *fsets_table, unsigned char *rules)

GaussParamsDtype = np.dtype([('mu', 'f4'), ('sigma', 'f4')])

@cython.boundscheck(False)
@cython.wraparound(False)
def tune_lfs_gpu(
    np.ndarray[unsigned, ndim=1, mode='c'] fsets_lens not None,
    np.ndarray[GaussParams, ndim=2] xx_train not None,
    np.ndarray[unsigned, ndim=1] y_train not None,
    *, rules_len=10, mu=100, lamda=500, iterations=500):

    if not xx_train.flags['C_CONTIGUOUS']:
        xx_train = np.ascontiguousarray(xx_train)
    if not y_train.flags['C_CONTIGUOUS']:
        y_train = np.ascontiguousarray(y_train)
    cdef unsigned N, n
    N, n = xx_train.shape[0], xx_train.shape[1]
    cdef np.ndarray[GaussParams, ndim=1, mode="c"] fsets_table = np.zeros(sum(fsets_lens), dtype=GaussParamsDtype)
    cdef np.ndarray[unsigned char, ndim=2, mode="c"] rules = np.zeros((rules_len, n+1), dtype=np.uint8)
    tune_lfs_gpu_impl(&fsets_lens[0], rules_len, n, &xx_train[0, 0], &y_train[0], N, mu, lamda, iterations, &fsets_table[0], &rules[0, 0])
    return fsets_lens, fsets_table, rules

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_gpu(
    np.ndarray[unsigned, ndim=1, mode='c'] fsets_lens not None,
    np.ndarray[GaussParams, ndim=1, mode='c'] fsets_table not None,
    np.ndarray[unsigned char, ndim=2, mode='c'] rules not None,
    np.ndarray[GaussParams, ndim=2, mode='c'] xxs not None):

    cdef unsigned N, n, rules_len
    N, n, rules_len = xxs.shape[0], rules.shape[1] - 1, rules.shape[0]
    cdef np.ndarray[unsigned, ndim=1, mode='c'] ys = np.zeros(N, dtype=np.uint32)
    predict_gpu_impl(&fsets_lens[0], &fsets_table[0], &rules[0, 0], rules_len, n, &xxs[0, 0], &ys[0], N)
    return ys
