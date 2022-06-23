from tools.distance.abstractDist import AbstractDist
import torch
import numpy as np

from typing import Tuple


class HeurEuc(AbstractDist):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def _calc_dist(
        self,
        x: AbstractDist._types,
        y: AbstractDist._types,
        feat_params_x: Tuple[float, float] = (0.0, 1.0),
        feat_params_y: Tuple[float, float] = (0.0, 1.0),
    ) -> float:
        _type = torch if isinstance(x, torch.Tensor) else np

        e_x, var_x = feat_params_x
        e_y, var_y = feat_params_y

        e_x_squared = var_x - e_x**2
        e_y_squared = var_y - e_y**2

        existence_x = _type.isnan(x)
        existence_y = _type.isnan(y)

        if _type == torch:
            existence_x = existence_x.long()
            existence_y = existence_y.long()

        else:
            existence_x = existence_x.astype(int)
            existence_y = existence_y.astype(int)

        existence = existence_x + 2 * existence_y

        res = _type.zeros_like(x)
        res = _type.where(existence == 0, (x - y) ** 2, res)
        res = _type.where(existence == 1, y**2 - 2 * y * e_x + e_x_squared, res)
        res = _type.where(existence == 2, x**2 - 2 * x * e_y + e_y_squared, res)
        res = _type.where(
            existence == 3,
            (e_x_squared + e_y_squared - 2 * e_x * e_y) * _type.ones_like(x),
            res,
        )

        res = res.sum() ** 0.5

        if _type == torch:
            res = res.item()

        return res

    @property
    def name(self):
        return "Heuristic Euclidian"


if __name__ == "__main__":
    from euclidian import EuclidianDist

    dist = EuclidianDist(ignore_nan=True)
    heur = HeurEuc(alpha=1.0, beta=1.0)
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)

    e_x, var_x = x.nanmean(axis=0), x.var(axis=0)

    x[0, 1] = torch.nan
    y[0, 1] = torch.nan
    x[0, 2] = torch.nan
    y[0, 3] = torch.nan
    y[0, 5] = torch.nan

    print(dist(x[0], y[0]))
    print(heur(x[0], y[0]))
    print(heur(x[0], y[0], feat_params_x=(e_x, var_x)))

