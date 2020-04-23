#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evotunf import LogicalFuzzyClassifier, GaussParamsDtype


DATA = pd.read_csv('datasets/balance-scale/balance-scale.data', names=[
    'cls', 'lw', 'ld', 'rw', 'rd'], dtype=str)

ATTR_DICT = OrderedDict([
    *[(c, list(map(str, range(1, 6)))) for c in ['lw', 'ld', 'rw', 'rd']],
    ('cls', ['L', 'B', 'R']),
])
ATTR_NAMES = list(ATTR_DICT.keys())

def _filter_dict_by_not_none_keys(d, keys=None):
    return type(d)([(k, d.get(k)) for k in (keys if keys is not None else d) if d.get(k) is not None ])

def build_fsets(categories):
    k = len(categories)
    return OrderedDict((c, ((i + 0.5) / k, 0.5 / k)) for i, c in enumerate(categories))

def build_xxs(data, attr_dict):
    attr2fsets_mapping = OrderedDict([(c, build_fsets(l)) for c, l in attr_dict.items()])
    return np.array([[c2f[row[c]] for c, c2f in attr2fsets_mapping.items()]
                     for _, row in data.iterrows()], dtype=GaussParamsDtype)

def build_ys(data, attr, categories):
    k = len(categories)
    category2idx = OrderedDict((c, i) for i, c in enumerate(categories))
    return np.array([category2idx[y] for y in data[attr]], dtype=np.uint32)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--strategy', choices=('cpu', 'gpu'), default='gpu')
parser.add_argument('-i', '--iterations', type=int)
parser.add_argument('-r', '--rules', type=int)
parser.add_argument('--mu', type=int)
parser.add_argument('--lamda', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    config = _filter_dict_by_not_none_keys(vars(args))
    print('Config:', config)

    xxs = build_xxs(DATA, _filter_dict_by_not_none_keys(ATTR_DICT, ATTR_NAMES[:-1]))
    ys = build_ys(DATA, ATTR_NAMES[-1], ATTR_DICT[ATTR_NAMES[-1]])
    fsets_lens = np.array([len(ATTR_DICT[a]) for a in ATTR_NAMES], dtype=np.uint32)
    # print(fsets_lens)
    xx_train, xx_test, y_train, y_test = train_test_split(xxs, ys, test_size=0.2, shuffle=False)

    # lfc = LogicalFuzzyClassifier().fit(fsets_lens, xx_train, y_train, strategy=strategy, iterations=5)
    lfc = LogicalFuzzyClassifier().fit(fsets_lens, xxs, ys, **config)
    print(lfc.score(xx_test, y_test, **_filter_dict_by_not_none_keys(config, ('strategy',))))
