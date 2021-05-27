#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evotunf import LogicalFuzzyClassifier, GaussParamsDtype
from parser import get_parser
from utils import _filter_dict_by_not_none_keys


DATA = pd.read_csv('datasets/balance-scale/balance-scale.data', names=[
    'cls', 'lw', 'ld', 'rw', 'rd'], dtype=str)

ATTR_DICT = OrderedDict([
    *[(c, list(map(str, range(1, 6)))) for c in ['lw', 'ld', 'rw', 'rd']],
    ('cls', ['L', 'B', 'R']),
])
ATTR_NAMES = list(ATTR_DICT.keys())

GaussParamsDtype = np.dtype([('mu', 'f4'), ('sigma', 'f4')])

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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = _filter_dict_by_not_none_keys(vars(args))
    print('Config:', config)

    xxs = build_xxs(DATA, _filter_dict_by_not_none_keys(ATTR_DICT, DATA.columns[:-1]))
    ys = build_ys(DATA, DATA.columns[-1], ATTR_DICT[DATA.columns[-1]])
    fsets_lens = np.array([len(ATTR_DICT[a]) for a in DATA.columns], dtype=np.uint32)
    # print(fsets_lens)
    xx_train, xx_test, y_train, y_test = train_test_split(xxs, ys, test_size=0.2,
                                                          shuffle=True, random_state=42)

    lfc = LogicalFuzzyClassifier(fsets_lens).fit(xx_train, y_train, **config, strategy='cpu')
    print(lfc.fsets)
    print(lfc.rules)
    print(y_test)
    print(lfc.score(xx_test, y_test, strategy='cpu'))
