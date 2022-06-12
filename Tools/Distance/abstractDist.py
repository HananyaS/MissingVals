import torch
import numpy as np
from abc import ABC, abstractmethod


class AbstractDist(ABC):
    _types = (torch.Tensor, np.ndarray)

    def __call__(self, x: _types, y: _types) -> float:
        return self.calc_dist(x, y)

    def calc_dist(self, x: _types, y: _types, *args, **kwargs) -> float:
        assert type(x) == type(y), "x and y must be of the same type"
        assert x.shape == y.shape, "x and y must have the same shape"

        return self._calc_dist(x, y, *args, **kwargs)

    @abstractmethod
    def _calc_dist(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} distance"

    def __repr__(self):
        return f"{self.name} distance"

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError
