from datasets.graphsDataset import GraphsDataset
from datasets.tabDataset import TabDataset
from pipeline.stage import Stage


class GraphsDataStage(
    Stage,
    _tasks=[
        "from_tab",
        "norm",
        "get_train",
        "get_test",
        "get_val",
    ],
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task == "from_tab":
            return self._from_tab(*args, **kwargs, **self.run_kwargs)

        if self.task == "norm":
            return self._norm(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_train":
            return self._get_train(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_test":
            return self._get_test(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_val":
            return self._get_val(*args, **kwargs, **self.run_kwargs)

        raise ValueError(f"Unknown task:\t{self.task}, Available tasks: {self._tasks}")

    @staticmethod
    def _from_tab(tab_data: TabDataset, **kwargs):
        return GraphsDataset.from_tab(tab_data=tab_data, **kwargs)

    @staticmethod
    def _norm(data: GraphsDataset):
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
        return f"Graphs Data Stage {self.id}\t{self.name}"
