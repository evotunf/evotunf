#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evotunf import LogicalFuzzyRegressor


# data = pd.read_csv(
#     'datasets/Wine/wine.data',
#     names=['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
#            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'],
#     dtype=np.float32)
data = pd.read_csv(
    'datasets/AirfoilSelfNoise/airfoil_self_noise.dat', sep='\t',
    names=['Frequency', 'Angle', 'Chord length', 'Velocity', 'Displacement thickness', 'Pressure'],
    dtype=np.float32)
data.info()

train, test = train_test_split(data, test_size=0.2)
xx_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
xx_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

lfr = LogicalFuzzyRegressor().fit(xx_train.to_numpy(), y_train.to_numpy(), strategy='cpu')
print(lfr.score(xx_test.to_numpy(), y_test.to_numpy()))
