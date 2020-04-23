import numpy as np

from evotunf_ext import tune_lfs_cpu, predict_cpu
from evotunf_ext import tune_lfs_gpu, predict_gpu


class LogicalFuzzyClassifier:
    def __init__(self):
        self.fsets_lens = None
        self.fsets_table = None
        self.rules = None

    def fit(self, fsets_lens, xx_train, y_train, *, strategy='cpu', rules=10, mu=10, lamda=50, iterations=500):
        assert strategy in ('cpu', 'gpu')
        if strategy == 'cpu':
            tune_lfs = tune_lfs_cpu
        elif strategy== 'gpu':
            tune_lfs = tune_lfs_gpu
        self.fsets_lens, self.fsets_table, self.rules = (
            tune_lfs(fsets_lens, xx_train, y_train, rules_len=rules, mu=mu, lamda=lamda, iterations=iterations))
        return self

    def predict(self, xxs, strategy='cpu'):
        assert strategy in ('cpu', 'gpu')
        if strategy == 'cpu':
            predict = predict_cpu
        elif strategy == 'gpu':
            predict = predict_gpu
        ys = predict(self.fsets_lens, self.fsets_table, self.rules, xxs)
        return ys

    def score(self, xxs, ys, strategy='cpu'):
        return np.sum(np.abs(self.predict(xxs, strategy) == ys)) / len(xxs)
