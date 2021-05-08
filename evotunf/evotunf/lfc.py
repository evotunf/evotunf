from time import time

import numpy as np

from evotunf_ext import (
    tune_lfs_cpu, predict_cpu,
    tune_lfs_gpu, predict_gpu)
from .fuzzy import predict as predict_py


class LogicalFuzzyClassifier:
    def __init__(self, fset_lens):
        self.fset_lens = np.array(fset_lens, dtype=np.uint32)
        self.fsets = None
        self.rules = None

    def fit(self, uxxs, ys, *, rules=10, population=50, iterations=500, strategy='gpu'):
        start = time()
        if strategy == 'gpu':
            self.fsets, self.rules = (
                tune_lfs_gpu(self.fset_lens, uxxs, ys,rules_len=rules,
                             population_power=population, iterations=iterations))
        elif strategy == 'cpu':
            self.fsets, self.rules = (
                tune_lfs_cpu(self.fset_lens, uxxs, ys, rules_len=rules,
                             population_power=population, iterations=iterations))
        print('Fit duration:', time() - start)
        return self

    def predict(self, uxxs, strategy='gpu'):
        f = lambda x: print(x) or x
        if strategy == 'gpu':
            return f(predict_gpu(self.fset_lens, self.fsets, self.rules, uxxs))
        elif strategy == 'cpu':
            return f(predict_cpu(self.fset_lens, self.fsets, self.rules, uxxs))
        elif strategy == 'py':
            return f(predict_py(self.fset_lens, self.v, self.fsets, self.rules, uxxs))

    def score(self, uxxs, ys, strategy='gpu'):
        pred = self.predict(uxxs, strategy)
        print(ys)
        print(pred)
        return np.sum(pred == ys) / len(uxxs)
