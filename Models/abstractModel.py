import torch
from abc import abstractmethod, ABC


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError
