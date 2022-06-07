import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from typing import Union

from sklearn.neighbors import NearestNeighbors


def hur_dist(
    s1_idx: int,
    s2_idx: int,
    data: Union[pd.DataFrame, np.ndarray],
    dims_to_ignore: Union[pd.Series, np.ndarray] = [],
    alpha: float = 1.0,
    beta: float = 1.0,
):
    if isinstance(dims_to_ignore, pd.Series):
        dims_to_ignore = dims_to_ignore.values

    s1, s2 = (
        data[s1_idx, list(set(range(data.shape[1])) - set(dims_to_ignore))],
        data[s2_idx, list(set(range(data.shape[1])) - set(dims_to_ignore))],
    )
    assert len(s1) == len(s2)

    if isinstance(s1, pd.Series):
        s1 = s1.values

    if isinstance(s2, pd.Series):
        s2 = s2.values

    if isinstance(data, pd.DataFrame):
        data = data.values

    s = 0

    for i in range(len(s1)):
        if s1[i] == s1[i] and s2[i] == s2[i]:
            s += (s1[i] - s2[i]) ** 2

        elif s1[i] == s1[i]:
            # # E[ (s1 - s2) ^ 2 | s1]
            # s += (s1[i]) ** 2
            s += (beta**2) * (s1[i] ** 2 + 1)
            """
            imputation_options = {}
            for e2 in existing_s2:
                similar_samples = [
                    data[j, i]
                    for j in range(data.shape[0])
                    if abs(data[j, e2] - s2[e2]) < 0.1 and data[j, i] == data[j, i]
                ]

                imputation_options[e2] = (
                    np.mean(similar_samples),
                    len(similar_samples),
                )

            if len(imputation_options) == 0:
                imputed_val = 0

            else:
                imputed_val, counter = max(
                    imputation_options.values(), key=lambda x: x[1]
                )

                if counter == 0:
                    imputed_val = 0

            s += (s1[i] - imputed_val) ** 2
            """

        elif s2[i] == s2[i]:
            # # E[ (s1 - s2) ^ 2 | s2]
            # s += abs(s2[i])
            s += (beta**2) * (s2[i] ** 2 + 1)

            """
            imputation_options = {}
            for e1 in existing_s1:
                similar_samples = [
                    data[j, i]
                    for j in range(data.shape[0])
                    if abs(data[j, e1] - s2[e1]) < 0.1 and data[j, i] == data[j, i]
                ]
                imputation_options[e1] = (
                    np.mean(similar_samples),
                    len(similar_samples),
                )

            if len(imputation_options) == 0:
                imputed_val = 0

            else:
                imputed_val, counter = max(
                    imputation_options.values(), key=lambda x: x[1]
                )

                if counter == 0:
                    imputed_val = 0

            # s += (s2[i] - imputed_val) ** 2
            s += abs(s2[i] - imputed_val)
            """

        # E[ (s1 - s2) ^ 2 ] = 2
        else:
            s += 2 * (alpha**2)

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

    knn(df, hur_dist)

    # for i, j in product(range(data.shape[0]), range(data.shape[1])):
    #     if np.random.binomial(1, 0.5) == 1:
    #         df.iloc[i, j] = np.nan
    #
    # print(dist(df.iloc[:, 0], df.iloc[:, 1], df.cov()))


def knn(df: Union[pd.DataFrame, np.ndarray], dist_func=hur_dist, include_self=False):
    if isinstance(df, pd.DataFrame):
        df = df.values

    all_dists = -1 * np.ones((df.shape[0], df.shape[0]))

    for i, s1 in enumerate(df):
        for j, s2 in enumerate(df):
            all_dists[i, j] = dist_func(s1, s2)


if __name__ == "__main__":
    test()
