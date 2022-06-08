import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from typing import Union

from sklearn.neighbors import NearestNeighbors


def dist(s1: Union[pd.Series, np.ndarray], s2: Union[pd.Series, np.ndarray]):
    assert len(s1) == len(s2)

    if isinstance(s1, pd.Series):
        s1 = s1.values

    if isinstance(s2, pd.Series):
        s2 = s2.values

    s = 0

    for i in range(len(s1)):
        if s1[i] == s1[i] and s2[i] == s2[i]:
            s += (s1[i] - s2[i]) ** 2

        elif s1[i] == s1[i]:
            # E[ (s1 - s2) ^ 2 | s1]
            s += s1[i]

        elif s2[i] == s2[i]:
            # E[ (s1 - s2) ^ 2 | s2]
            s += s2[i]

        # E[ (s1 - s2) ^ 2 ] = 2
        else:
            s += 2

    return s**0.5


def test_E():
    d = []
    n = 1000000
    x = np.random.standard_normal(n)
    y = np.random.standard_normal(n)
    d = x - y
    d_2 = (x - y) ** 2

    print(f"Mean diff:\t{np.round(np.mean(d), 4)}")
    print(f"Mean squared diff:\t{np.round(np.mean(d_2), 4)}")

    plt.hist(d, color="blue", edgecolor="black", bins=int(180 / 5))
    plt.show()

    plt.hist(d_2, color="blue", edgecolor="black", bins=int(180 / 5))
    plt.show()


def test():
    data = np.random.standard_normal((1000, 10))
    df = pd.DataFrame(data)

    knn(df, dist)

    # for i, j in product(range(data.shape[0]), range(data.shape[1])):
    #     if np.random.binomial(1, 0.5) == 1:
    #         df.iloc[i, j] = np.nan
    #
    # print(dist(df.iloc[:, 0], df.iloc[:, 1], df.cov()))


def knn(df: Union[pd.DataFrame, np.ndarray], dist_func=dist, include_self=False):
    if isinstance(df, pd.DataFrame):
        df = df.values

    all_dists = -1 * np.ones((df.shape[0], df.shape[0]))

    for i, s1 in enumerate(df):
        for j, s2 in enumerate(df):
            all_dists[i, j] = dist_func(s1, s2)



if __name__ == "__main__":
    test()
