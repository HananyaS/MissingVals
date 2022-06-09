import xgboost as xgb
from abstractModel import AbstractModel as AbstractModel

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
            return self.clf.predict_proba(test_loader.dataset.get_features())

        return self.clf.predict(test_loader.dataset.get_features())

    def evaluate(self, metric: str = "auc", **kwargs) -> float:
        if metric == "auc":
            return self._eval_auc(**kwargs)

        raise NotImplementedError

    def _eval_auc(
        self,
        loader: DataLoader,
        labels_from_loader: Callable = lambda loader: loader.dataset.get_labels(),
    ) -> float:
        self.eval()
        preds = self.predict(loader, probs=True)
        labels = labels_from_loader(loader)

        return roc_auc_score(labels, preds)
