from .stage import Stage
from typing import List, Any
from copy import copy


class Pipeline:
    def __init__(self, name: str, *stages: Stage):
        self.name = name
        self.n_stages = len(stages)
        self.stages = Pipeline.make_pipline(stages)

    @staticmethod
    def make_pipline(stages: List[Stage]) -> List[Stage]:
        _stages = []

        for i, stage in enumerate(stages):
            _stages.append(copy(stage).set_id(i))

        return _stages

    def _make_pipline(self, *stages: Stage):
        pipe = Pipeline.make_pipline(*stages)
        self.n_stages = len(pipe)
        return pipe

    def add_stage(self, stage: Stage):
        self.stages.append(stage)

    def get_num_stages(self):
        return self.n_stages

    def __add__(self, other):
        assert isinstance(
            other, Pipeline
        ), "Pipeline can only be added to another Pipeline"
        return Pipeline(f"{self.name} -> {other.name}", *self.stages, *other.stages)

    def run(self, _input: Any, verbose: bool = False):
        res = _input
        for i, stage in enumerate(self.stages):
            try:
                if verbose:
                    print(f"Stage {i} started")

                res = stage(res)

            except Exception as e:
                print(f"{stage} failed with error: {e}")
                break

        return res

    def __getitem__(self, idx: int):
        assert idx in range(len(self.stages)), "Index out of range"
        return self.stages[idx]

    @classmethod
    def empty_pipeline(cls, name: str = "Empty"):
        return cls(name, Stage.empty())

    @property
    def empty(self):
        return len(self.stages) == 0

    def __str__(self):
        if self.empty:
            return f"Empty Pipeline {self.name}"
        return f"Pipeline {self.name}:\t{' -> '.join([stage.name for stage in self.stages])}"

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


if __name__ == "__main__":
    pipe1 = Pipeline.empty_pipeline("Test 1")
    pipe2 = Pipeline.empty_pipeline("Test 2")
    pipe = pipe1 + pipe2
    x = [1, 2, 3]
    print(pipe(x))
