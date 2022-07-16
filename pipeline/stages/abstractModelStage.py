from pipeline.stage import Stage
from abc import ABC

# from pipeline.addTaskMeta import AddTasksMeta


class AbstractModelStage(Stage, ABC, _tasks=["fit", "predict", "evaluate", "build"]):
    # _additional_tasks = ["fit", "predict", "evaluate", "build"]

    def __init__(self, **kwargs):
        # self._tasks = self._tasks + self._additional_tasks
        super().__init__(**kwargs)

    def __str__(self):
        return f"Model Stage {self.id}\t{self.name}"
