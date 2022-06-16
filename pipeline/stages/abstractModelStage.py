from pipeline.stage import Stage
from abc import ABC


class AbstractModelStage(Stage, ABC):
    _additional_tasks = ["fit", "predict", "evaluate", "build"]

    def __init__(
        self,
        **kwargs,
    ):
        self._tasks.extend(self._additional_tasks)
        super().__init__(**kwargs)

    def __str__(self):
        return f"ModelStage {self.id}\t{self.name}"
