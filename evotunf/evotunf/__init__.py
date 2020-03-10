print("=== Hello from EvoTunF =:)==")


import numpy as np

from evotunf_ext import train_lfs_cpu, predict_lfs_cpu


class LogicalFuzzyRegressor:
    def __init__(self):
        self.t_outer_kind = None
        self.t_inner_kind = None
        self.impl_kind = None
        self.rule_base = None

    def fit(self, xx_train, y_train, *, strategy='cpu'):
        assert strategy in ('cpu',)
        if strategy == 'cpu':
            (self.t_outer_kind, self.t_inner_kind, self.impl_kind, self.rule_base) = (
                    train_lfs_cpu(xx_train, y_train))
        return self

    def predict(self, xxs, strategy='cpu'):
        assert strategy in ('cpu',)
        if strategy == 'cpu':
            ys = predict_lfs_cpu(self.t_outer_kind, self.t_inner_kind, self.impl_kind, self.rule_base, xxs)
        return ys

    def score(self, xxs, ys, strategy='cpu'):
        return np.sum(np.abs(self.predict(xxs, strategy) - ys)) / len(xxs)
