#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evotunf import LogicalFuzzyClassifier


DATA = pd.read_csv('datasets/car/car.data', names=[
    'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
# DATA.info()

ATTR_DICT = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high'],
    'class': ['unacc', 'acc', 'good', 'vgood'],
}

GaussParamsDtype = np.dtype([('mu', 'f4'), ('sigma', 'f4')])

def _filter_dict_by_not_none_keys(d, keys=None):
    return {k: d.get(k) for k in (keys if keys is not None else d) if d.get(k) is not None}

def build_fsets(categories):
    k = len(categories)
    return OrderedDict((c, ((i + 0.5) / k, 0.5 / k)) for i, c in enumerate(categories))

def build_xxs(data, attr_dict):
    attr2fsets_mapping = {c: build_fsets(l) for c, l in attr_dict.items()}
    return np.array([[c2f[xx[c]] for c, c2f in attr2fsets_mapping.items()]
                     for _, xx in data.iterrows()], dtype=GaussParamsDtype)

def build_ys(data, attr, categories):
    k = len(categories)
    category2idx = OrderedDict((c, i) for i, c in enumerate(categories))
    return np.array([category2idx[y] for y in data[attr]], dtype=np.uint32)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', type=int)
parser.add_argument('-r', '--rules', type=int)
parser.add_argument('--mu', type=int)
parser.add_argument('--lamda', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    config = _filter_dict_by_not_none_keys(vars(args))
    print('Config:', config)

    xxs = build_xxs(DATA, _filter_dict_by_not_none_keys(ATTR_DICT, DATA.columns[:-1]))
    ys = build_ys(DATA, DATA.columns[-1], ATTR_DICT[DATA.columns[-1]])
    fsets_lens = np.array([len(ATTR_DICT[a]) for a in DATA.columns], dtype=np.uint32)
    # print(fsets_lens)
    xx_train, xx_test, y_train, y_test = train_test_split(xxs, ys, test_size=0.2, shuffle=False)

    # lfc = LogicalFuzzyClassifier().fit(fsets_lens, xx_train, y_train, strategy=strategy, iterations=5)
    lfc = LogicalFuzzyClassifier().fit(fsets_lens, xx_train, y_train, **config)
    print(lfc.score(xx_test, y_test))
