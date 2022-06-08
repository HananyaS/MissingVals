import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Union, List, Tuple


class TabDataPair:
    _types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        X: _types,
        dtype: _types = torch.Tensor,
        name: str = "",
        Y: _types = None,
        normalize: bool = False,
        normalization_params: Tuple[List, List] = None,
    ):
        assert Y is None or X.shape[0] == Y.shape[0]
        self.dtype = dtype
        self.X = self._transform_types(X)
        self.Y = self._transform_types(Y)

        if normalize:
            self._zscore(normalization_params)

        self.name = name
        self.train = self.Y is not None

    def _transform_types(self, data: _types):
        np2torch = lambda x: torch.from_numpy(x).float()
        torch2np = lambda x: x.cpu().detach().numpy()

        if type(data) == self.dtype:
            return data.astype(float) if self.dtype == np.ndarray else data.float()

        if type(data) == np.ndarray:
            return np2torch(data)

        return torch2np(data)

    def _zscore(self, params: Tuple[List, List] = None):
        if params is not None:
            mu, sigma = params
            assert len(mu) == len(sigma) == self.X.shape[0]

        else:
            mu, sigma = self.X.mean(axis=0), self.X.std(axis=0)

        self.X = (self.X - mu) / sigma

        return self

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]

    def __str__(self):
        return f'Dataset {self.name}; length {len(self)}; mode {"train" if self.train else "test"}'

"""
class TabDataset:
    _types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        train_X: _types,
        train_Y: _types = None,
        test_X: _types = None,
        test_Y: _types = None,
        val_X: _types = None,
        val_Y: _types = None,
    ):
        ...
"""

def tdp_test():
    x = torch.randint(100, (100, 10))
    # x = np.random.standard_normal(100)
    y = torch.ones((100,))

    tdp = TabDataPair(X=x, Y=y, name="Check", dtype=torch.Tensor, normalize=True)
    print(tdp)
