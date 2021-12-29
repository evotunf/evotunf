#!/usr/bin/env python
import io

import numpy as np
import pandas as pd

from evotunf import LogicalFuzzyClassifier, GaussParamsDtype
from parser import get_parser
from utils import _filter_dict_by_not_none_keys


GaussParamsDtype = np.dtype([('mu', 'f4'), ('sigma', 'f4')])

def build_uxxs(xxs, fset_lens):
    return np.array([[(float(x) / (n+1), 1.0 / (n+1))
                      for i, (x, n) in enumerate(zip(xx, fset_lens))]
                     for xx in xxs], dtype=GaussParamsDtype)


samples = np.array([
    #[2, 4, 2, 2, 4, 2, 2]
    # *,*,*,*,*,2,2
    [1,1,1,1,1,2,2],
    [1,2,2,1,4,1,2],
    [1,3,1,2,3,2,2],
    [2,4,2,2,4,2,2],
    # 2,*,*,*,*,1,1
    [2,1,1,1,1,1,1],
    [2,3,1,2,3,1,1],
    [2,4,2,2,4,1,1],
    # 1,2,*,*,*,1,1
    [1,2,1,1,1,1,1],
    # 1,1,*,*,*,1,1
    [1,1,1,1,1,1,1],
    # 1,3,2,2,*,1,1
    [1,3,2,2,1,1,1],
    [1,2,2,2,3,1,1],
    # *,*,*,*,4,1,1
    [1,1,1,1,4,1,1],
    # 1,4,*,*,1,1,2
    [1,4,1,1,1,1,2],
    # 1,4,*,*,2,1,2
    [1,4,1,1,2,1,2],
    # 1,4,*,*,3,1,2
    [1,4,1,1,3,1,2],
    # 1,3,1,1,1,1,2
    [1,3,1,1,1,1,2],
    # 1,3,1,1,2,1,2
    [1,3,1,1,2,1,2],
    # 1,3,1,2,1,1,2
    [1,3,1,2,1,1,2],
    # 1,3,1,2,2,1,2
    [1,3,1,2,2,1,2],
    # 1,3,1,1,3,1,1
    [1,3,1,1,3,1,1],
    # 1,3,1,2,3,1,2
    [1,3,1,2,3,1,2],
])
fset_lens = [2, 4, 2, 2, 4, 2, 2]


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = _filter_dict_by_not_none_keys(vars(args))
    print('Config:', config)

    xxs, ys = np.hsplit(samples, [-1])
    uxxs = build_uxxs(xxs, fset_lens)
    ys = ys.reshape(-1).astype(np.uint32)

    lfc = LogicalFuzzyClassifier(fset_lens=fset_lens)
    lfc.fit(uxxs, ys, strategy='gpu', **config)
#     lfc.fsets = np.array([
#         (0.3333333333333333, 0.3333333333333333), (0.6666666666666666, 0.3333333333333333),
#         (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
#         (0.3333333333333333, 0.3333333333333333), (0.6666666666666666, 0.3333333333333333),
#         (0.3333333333333333, 0.3333333333333333), (0.6666666666666666, 0.3333333333333333),
#         (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
#         (0.3333333333333333, 0.3333333333333333), (0.6666666666666666, 0.3333333333333333),
#         (0.3333333333333333, 0.3333333333333333), (0.6666666666666666, 0.3333333333333333),
#     ], dtype=GaussParamsDtype)
#     lfc.rules = np.array([
# [0,0,0,0,0,2,2],
# [2,0,0,0,0,1,1],
# [1,2,0,0,0,1,1],
# [1,1,0,0,0,1,1],
# [1,3,2,2,0,1,1],
# [0,0,0,0,4,1,1],
# [1,4,0,0,1,1,2],
# [1,4,0,0,2,1,2],
# [1,4,0,0,3,1,2],
# [1,3,1,1,1,1,2],
# [1,3,1,1,2,1,2],
# [1,3,1,2,1,1,2],
# [1,3,1,2,2,1,2],
# [1,3,1,1,3,1,1],
# [1,3,1,2,3,1,2],
#     ], dtype=np.int8)
    print(lfc.fsets)
    print(np.sort(lfc.rules, axis=0))
    lfc.predict(uxxs, strategy='gpu')
    print(lfc.score(uxxs, ys, strategy='gpu'))
    
