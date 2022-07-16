import torch
from datasets.tabDataset import TabDataset
from pipeline.stage import Stage
from typing import Type


class TabDataStage(
    Stage,
    _tasks=[
        "load",
        "norm",
        "get_train",
        "get_test",
        "get_val",
        "setX",
    ],
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task == "load":
            return self._load(*args, **kwargs, **self.run_kwargs)

        if self.task == "norm":
            return self._norm(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_train":
            return self._get_train(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_test":
            return self._get_test(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_val":
            return self._get_val(*args, **kwargs, **self.run_kwargs)

        if self.task == "setX":
            return self._set_X(*args, **kwargs, **self.run_kwargs)

        raise ValueError(f"Unknown task:\t{self.task}, Available tasks: {self._tasks}")

    @staticmethod
    def _load(data_dir: str, **kwargs):
        return TabDataset.load(data_dir=data_dir, **kwargs)

    @staticmethod
    def _norm(data: TabDataset):
        return data.zscore()

    @staticmethod
    def _get_train(data: TabDataset, **kwargs):
        return data.get_train_data(**kwargs)

    @staticmethod
    def _get_test(data: TabDataset, **kwargs):
        return data.get_test_data(**kwargs)

    @staticmethod
    def _get_val(data: TabDataset, **kwargs):
        return data.get_val_data(**kwargs)

    @staticmethod
    def _set_X(data: TabDataset, set_: str, X: Type[torch.Tensor]):
        assert set_ in ["train", "test", "val"]

        if set_ == "train":
            data.train.X = X

        elif set_ == "val":
            assert data.val_exists, "Val set doesn't exist"
            data.val.X = X
        else:
            assert data.test_exists, "Test set doesn't exist"
            data.test.X = X

        data.normalized = False

        return data

    def __str__(self):
        return f"Tabular Data Stage {self.id}\t{self.name}"
