from datasets.tabDataset import TabDataset
from pipeline.stage import Stage
from typing import List


class DataStage(Stage):
    _tasks: List[str] = ["load", "norm", "get_train", "get_test", "get_val"]

    def __init__(self, task: str, **kwargs):
        assert task in self._tasks, f"Unknown task:\t{task}"
        super().__init__(task=task, **kwargs)

    def run(self, input_: str):
        if self.task == "load":
            return self._load(input_, **self.run_kwargs)

        if self.task == "norm":
            return self._norm(input_, **self.run_kwargs)

        if self.task == "get_train":
            return self._get_train(input_, **self.run_kwargs)

        if self.task == "get_test":
            return self._get_test(input_, **self.run_kwargs)

        if self.task == "get_val":
            return self._get_val(input_, **self.run_kwargs)

        raise ValueError(f"Unknown task:\t{self.task}")

    @property
    def tasks(self):
        return self._tasks

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

    def __str__(self):
        return f"DataStage {self.id}\t{self.name}"
