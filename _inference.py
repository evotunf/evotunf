import math
import io

import numpy as np
import pandas as pd


def gauss(mu, sigma, x):
    return math.exp(-pow((x - mu)/sigma, 2))


def impl(a, b):
    return min(1, 1-a+b)  # Lukaszewicz


def infer(fsets, rules, xx):

    def gauss_ex(i, j, x):
        return gauss(*fsets[i][j], x)

    num, den = 0.0, 0.0
    for b in rules[:, -1]: # b in [0, len(fsets[-1])
        uy_center, _ = fsets[-1][b-1]
        cross = 1.0
        for *aa, b in rules: # a_i in 0, [1, len(fsets[i])]
            cross = min(cross, max(
                min(gauss_ex(i, x-1, t), impl(gauss_ex(i, a-1, t), gauss_ex(-1, b-1, uy_center)))
                for i, (a, x) in enumerate(zip(aa, xx)) if a != 0
                for t in np.arange(0, 1.01, 0.05)
            ))
        num += uy_center * cross
        den += cross
    return max(range(len(fsets[-1])),
               key=lambda j: gauss_ex(-1, j, num / den))+1


fsets_lens = [2, 4, 2, 2, 4, 2, 2]
fsets = [
    [(i / (n+1), 1.0 / (n+1)) for i in range(1, n+1)]
    for n in fsets_lens
]
# print(fsets)


# Rules taken from Shuttle Landing Control Data Set:
# https://archive.ics.uci.edu/ml/datasets/Shuttle+Landing+Control
rules_str = """
2,*,*,*,*,*,2
1,2,*,*,*,*,1
1,1,2,*,*,*,1
1,1,1,*,*,*,1
1,1,3,2,2,*,1
1,*,*,*,*,4,1
2,1,4,*,*,1,1
2,1,4,*,*,2,1
2,1,4,*,*,3,1
2,1,3,1,1,1,1
2,1,3,1,1,2,1
2,1,3,1,2,1,1
2,1,3,1,2,2,1
1,1,3,1,1,3,1
2,1,3,1,2,3,1
""".replace('*', '0')
rules_df = pd.read_csv(io.StringIO(rules_str), names=[
    'cls', 'stability', 'error', 'sign',
    'wind', 'magnitude', 'visibility'])
rules = rules_df[['stability', 'error', 'sign', 'wind',
                  'magnitude', 'visibility', 'cls']].values


samples = [
    [1,1,1,1,1,2,2],
    [2,1,1,1,1,1,1],
    [1,2,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,3,2,2,1,1,1],
    [1,1,1,1,4,1,1],
    [1,4,1,1,1,1,2],
    [1,4,1,1,2,1,2],
    [1,4,1,1,3,1,2],
    [1,3,1,1,1,1,2],
    [1,3,1,1,2,1,2],
    [1,3,1,2,1,1,2],
    [1,3,1,2,2,1,2],
    [1,3,1,1,3,1,1],
    [1,3,1,2,3,1,2],
]


for *xx, y in samples:
    print(xx, y, infer(fsets, rules, xx))
            
