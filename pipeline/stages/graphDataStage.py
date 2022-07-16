from datasets.graphDataset import GraphDataset
from datasets.tabDataset import TabDataset
from pipeline.stage import Stage


class GraphDataStage(
    Stage,
    _tasks=[
        "from_tab",
        "norm",
        "get_train_loader",
        "get_test_loader",
        "get_val_loader",
    ],
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task == "from_tab":
            return self._from_tab(*args, **kwargs, **self.run_kwargs)

        if self.task == "norm":
            return self._norm(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_train_loader":
            return self._get_train_loader(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_test_loader":
            return self._get_test_loader(*args, **kwargs, **self.run_kwargs)

        if self.task == "get_val_loader":
            return self._get_val_loader(*args, **kwargs, **self.run_kwargs)

        raise ValueError(f"Unknown task:\t{self.task}, Available tasks: {self._tasks}")

    @staticmethod
    def _from_tab(tab_data: TabDataset, **kwargs):
        return GraphDataset.from_tab(tab_data=tab_data, **kwargs)

    @staticmethod
    def _norm(data: GraphDataset):
        return data.zscore()

    @staticmethod
    def _get_train_loader(data: GraphDataset):
        return data.get_train_loader()

    @staticmethod
    def _get_val_loader(data: GraphDataset):
        return data.get_val_loader()

    @staticmethod
    def _get_test_loader(data: GraphDataset):
        return data.get_test_loader()

    def __str__(self):
        return f"Graph Data Stage {self.id}\t{self.name}"
