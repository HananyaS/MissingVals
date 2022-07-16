import numpy as np
import torch

from tools.gfp import GFP
from datasets.tabDataPair import TabDataPair
from datasets.tabDataset import TabDataset

from pipeline.stage import Stage
from typing import List, Union

from tools.knn import KNN


class GFPStage(Stage, _tasks=["build", "prop"]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task == "build":
            return self._build(**kwargs, **self.run_kwargs)

        if self.task == "prop":
            return self._prop(*args, **kwargs, **self.run_kwargs)

    @staticmethod
    def _build(**model_kwargs):
        return GFP(**model_kwargs)

    @staticmethod
    def _prop(gfp_obj: GFP, data: Union[TabDataPair, TabDataset], **kwargs):
        if isinstance(data, TabDataPair):
            GFPStage._prop_tdp(gfp_obj, data, inplace=True, **kwargs)
            return data

        GFPStage._prop_tdp(gfp_obj, data.train, inplace=True, **kwargs)
        if data.test_exists:
            GFPStage._prop_tdp(gfp_obj, data.test, inplace=True, **kwargs)
        if data.val_exists:
            GFPStage._prop_tdp(gfp_obj, data.val, inplace=True, **kwargs)

        return data

    @staticmethod
    def _prop_tdp(
        gfp_obj: GFP,
        data: TabDataPair,
        inplace: bool = True,
        edges: Union[torch.Tensor, np.ndarray] = None,
        **kwargs
    ):
        if edges is None:
            knn = KNN(**kwargs)
            edges = knn.get_edges(data.X, return_type=torch.Tensor, as_adj=True)

        X_ = gfp_obj.prop(data.X, edges)

        if inplace:
            data.set_X(X_)

        return X_
