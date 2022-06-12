from abstractDist import AbstractDist
import torch
import numpy as np


class EuclidianDist(AbstractDist):
    def __init__(self, ignore_nan: bool = False):
        super().__init__()
        self.ignore_nan = ignore_nan

    def _calc_dist(self, x: AbstractDist._types, y: AbstractDist._types) -> float:
        _type = torch if isinstance(x, torch.Tensor) else np

        if self.ignore_nan:
            res = ((_type.nan_to_num(x) - _type.nan_to_num(y)) ** 2).sum() ** 0.5

        else:
            res = ((x - y) ** 2).sum() ** 0.5

            if res != res:
                raise ValueError("NaN detected")

        if _type == torch:
            res = res.item()

        return res

    @property
    def name(self):
        return "Euclidian"


dist = EuclidianDist(ignore_nan=True)
x = torch.Tensor([1, 2, 3])
y = torch.Tensor([3, 2, 1])

d = dist(x, y)
print(d)
print(dist)
