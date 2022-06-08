import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Union


class TabDataPair:
    _types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        X: _types,
        dtype: _types = torch.Tensor,
        Y: _types = None,
        name: str = None,
    ):
        self.dtype = dtype
        self.X = X
        self.Y = Y
        self.name = name
        self.train = self.Y is not None

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


X = torch.ones(100)
Y = X
tab_dp = TabDataPair(X=X, Y=Y, dtype=torch.Tensor)
print(tab_dp)
