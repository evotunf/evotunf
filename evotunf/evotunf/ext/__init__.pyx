import cython

import numpy as np
cimport numpy as np

cdef extern from "evotunf_ext.h":
    ctypedef struct LfsConfig:
        int t_outer
        int t_inner
        int impl
        unsigned rule_base_len
        float *gauss_params
    LfsConfig train_lfs_cpu_impl(float *xxs, float *ys, unsigned n, unsigned N, float *rule_base, unsigned rule_base_len)
    void infer_lfs_cpu_impl(LfsConfig lfs_config, float *xxs, float *ys, unsigned n, unsigned N)


@cython.boundscheck(False)
@cython.wraparound(False)
def train_lfs_cpu(
    np.ndarray[float, ndim=2] xx_train not None,
    np.ndarray[float, ndim=1] y_train not None):

    if not xx_train.flags['C_CONTIGUOUS']:
        xx_train = np.ascontiguousarray(xx_train)
    if not y_train.flags['C_CONTIGUOUS']:
        y_train = np.ascontiguousarray(y_train)
    cdef unsigned N, n, rule_base_len
    N, n = xx_train.shape[0], xx_train.shape[1]
    rule_base_len = 30
    cdef np.ndarray[float, ndim=3, mode='c'] rule_base = np.zeros((rule_base_len, n + 1, 2), dtype=np.float32)
    cdef LfsConfig lfs_config
    lfs_config = train_lfs_cpu_impl(&xx_train[0, 0], &y_train[0], n, N, &rule_base[0, 0, 0], rule_base_len)
    return (lfs_config.t_outer, lfs_config.t_inner, lfs_config.impl, rule_base)

@cython.boundscheck(False)
@cython.wraparound(False)
def predict_lfs_cpu(
    t_outer, t_inner, impl,
    np.ndarray[float, ndim=3] rule_base not None,
    np.ndarray[float, ndim=2] xxs not None):

    print(t_outer, t_inner, impl, (rule_base.shape[0], rule_base.shape[1], rule_base.shape[2]), rule_base)
    assert rule_base.shape[1] - 1 == xxs.shape[1], \
            f"The fuzzy system can accept only {rule_base.shape[1] - 1} inputs, but {xxs.shape[1]} was given."
    if not xxs.flags['C_CONTIGUOUS']:
        xxs = np.ascontiguousarray(xxs)
    cdef unsigned N, n, rule_base_len
    rule_base_len, n = rule_base.shape[0], rule_base.shape[1] - 1
    N = xxs.shape[0]
    cdef np.ndarray[float, ndim=1, mode='c'] ys = np.zeros((N,), dtype=np.float32)
    cdef LfsConfig lfs_config
    lfs_config.t_outer = t_outer
    lfs_config.t_inner = t_inner
    lfs_config.impl = impl
    lfs_config.rule_base_len = rule_base_len
    lfs_config.gauss_params = &rule_base[0, 0, 0]
    infer_lfs_cpu_impl(lfs_config, &xxs[0, 0], &ys[0], n, N)
    return ys
