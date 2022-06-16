from .stage import Stage
from typing import List
from copy import deepcopy


class Pipeline:
    def __init__(self, name: str, stages: List[Stage], verbose: bool = False):
        self.name = name
        self.n_stages = len(stages)
        self.stages = Pipeline.make_pipline(stages)
        self.cache = {}
        self.verbose = verbose

    @staticmethod
    def make_pipline(stages: List[Stage]) -> List[Stage]:
        _stages = []

        for i, stage in enumerate(stages):
            _stages.append(deepcopy(stage).set_id(i))

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

    def run(self, *init_args, **init_kwargs):
        # def run(self, _input: Any, verbose: bool = False):
        #     res = _input

        is_first = True
        for i, stage in enumerate(self.stages):
            try:
                if self.verbose:
                    print(f"Stage {i} started")

                if is_first:
                    res = stage(*init_args, **init_kwargs)
                    is_first = False

                elif stage.input_from is not None:
                    for var in stage.input_from.values():
                        assert var in self.cache.keys(), f'Var "{var}" not found in cache'

                    input_ = {param: self.cache[var] for param, var in stage.input_from.items()}
                    res = stage(**input_)

                else:
                    res = stage(res)

                if stage.store_in is not None:
                    assert (
                        stage.store_in not in self.cache.keys()
                    ), f"Var {stage.store_in} already in cache"
                    self.cache[stage.store_in] = res

            except Exception as e:
                raise Exception(f"{stage} failed with error: {e}")

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
