from pipeline.stages.abstractModelStage import AbstractModelStage
from models.abstractModel import AbstractModel
from torch.utils.data import DataLoader
from typing import Type


class MLModelStage(AbstractModelStage, _tasks=[]):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task == "build":
            return self._build(*args, **kwargs, **self.run_kwargs)

        if self.task == "fit":
            return self._fit(*args, **kwargs, **self.run_kwargs)

        if self.task == "predict":
            return self._predict(*args, **kwargs, **self.run_kwargs)

        if self.task == "evaluate":
            return self._evaluate(*args, **kwargs, **self.run_kwargs)

        raise ValueError(f"Unknown task:\t{self.task}, Available tasks: {self._tasks}")

    @staticmethod
    def _build(model_type: Type[AbstractModel], **model_kwargs):
        return model_type(**model_kwargs)

    @staticmethod
    def _fit(model: AbstractModel, train_loader: DataLoader, **kwargs):
        model.fit(train_loader=train_loader, **kwargs)
        return model

    @staticmethod
    def _predict(model: AbstractModel, test_loader: DataLoader, **kwargs):
        return model.predict(test_loader=test_loader, **kwargs)

    @staticmethod
    def _evaluate(model: AbstractModel, loader: DataLoader, **kwargs):
        return model.evaluate(loader=loader, **kwargs)

    def __str__(self):
        return f"ML Model Stage {self.id}\t{self.name}"
