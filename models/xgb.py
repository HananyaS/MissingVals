import xgboost as xgb
from .abstractModel import AbstractModel as AbstractModel

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from typing import Callable


class XGBoost(AbstractModel):
    def __init__(self, **params):
        self.clf = xgb.XGBClassifier(**params)

    def fit(self, train_loader: DataLoader):
        self.clf.fit(
            train_loader.dataset.get_features(), train_loader.dataset.get_labels()
        )

    def predict(self, test_loader: DataLoader, probs: bool = False):
        if probs:
            return torch.from_numpy(
                self.clf.predict_proba(test_loader.dataset.get_features())[:, 1]
            )

        return torch.from_numpy(self.clf.predict(test_loader.dataset.get_features()))

    def evaluate(self, metric: str = "auc", verbose=True, **kwargs) -> float:
        if metric == "auc":
            res = self._eval_auc(**kwargs)

        else:
            raise Exception(f"Unknown metric: {metric}")

        print(f"{metric.upper()} score is:\t{res:.4f}")
        return res

    def _eval_auc(
        self,
        val_loader: DataLoader,
        labels_from_loader: Callable = lambda loader: loader.dataset.get_labels(),
    ) -> float:
        preds = self.predict(val_loader, probs=True)
        labels = labels_from_loader(val_loader)

        return roc_auc_score(labels, preds)

    def __str__(self):
        return "XGBoost"
