import numpy as np

from evotunf_ext import tune_lfs_gpu, predict_gpu


class LogicalFuzzyClassifier:
    def __init__(self):
        self.fsets_lens = None
        self.fsets_table = None
        self.rules = None

    def fit(self, fsets_lens, xx_train, y_train, *, rules=10, mu=10, lamda=50, iterations=500):
        self.fsets_lens, self.fsets_table, self.rules = (
            tune_lfs_gpu(fsets_lens, xx_train, y_train, rules_len=rules, mu=mu, lamda=lamda, iterations=iterations))
        return self

    def predict(self, xxs):
        ys = predict_gpu(self.fsets_lens, self.fsets_table, self.rules, xxs)
        return ys

    def score(self, xxs, ys):
        return np.sum(np.abs(self.predict(xxs) == ys)) / len(xxs)
